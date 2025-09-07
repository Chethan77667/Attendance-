import os
import sys
import time
from datetime import datetime
from queue import Queue, Empty

import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from main import DeepFaceAttendance  # noqa: E402


# --- Helpers for auto-reloading data when files change ---
def _get_file_mtime(path: str) -> float:
    """Return last modified time for a file or 0.0 if missing."""
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0
    except Exception:
        return 0.0


RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


def get_system() -> DeepFaceAttendance:
    if "system" not in st.session_state:
        st.session_state.system = DeepFaceAttendance()
        # Track file mtimes for automatic reloads across pages/processors
        sys = st.session_state.system
        sys._students_mtime = _get_file_mtime(sys.data_file)
        sys._embeddings_mtime = _get_file_mtime(sys.embeddings_file)
    return st.session_state.system


class TakeAttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.system = get_system()
        self.last_ts = 0.0
        self.interval = 0.3
        self.queue = Queue(maxsize=100)
        self.seen = set()

        # Auto-reload settings
        self.last_reload_check = 0.0
        self.reload_check_interval = 1.5  # seconds

    def _maybe_reload_system(self) -> None:
        """Reload students/embeddings if their files changed while running.

        This allows immediate reflection of adds/deletes without restarting.
        """
        now = time.time()
        if now - self.last_reload_check < self.reload_check_interval:
            return
        self.last_reload_check = now

        sys = self.system
        try:
            students_mtime = _get_file_mtime(sys.data_file)
            embeddings_mtime = _get_file_mtime(sys.embeddings_file)

            changed = (
                students_mtime != getattr(sys, "_students_mtime", 0.0)
                or embeddings_mtime != getattr(sys, "_embeddings_mtime", 0.0)
            )
            if changed:
                sys.load_data()
                sys._students_mtime = students_mtime
                sys._embeddings_mtime = embeddings_mtime
                # Clear session caches so UI reflects changes instantly
                self.seen.clear()
                # Also clear pending items referencing deleted students
                try:
                    while not self.queue.empty():
                        item = self.queue.get_nowait()
                        if item and item.get("student_id") in sys.students:
                            # push back valid items (best-effort)
                            self.queue.put_nowait(item)
                            break
                except Exception:
                    pass
        except Exception:
            # Non-fatal; continue processing frames
            pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Apply hot-reload check
        self._maybe_reload_system()
        now = time.time()
        faces = self.system.detect_faces(img)
        if now - self.last_ts >= self.interval:
            for f in faces:
                x1, y1, x2, y2 = f['bbox']
                emb = f.get('embedding')
                if emb is None:
                    face_img = img[y1:y2, x1:x2]
                    if face_img.size > 0:
                        emb = self.system.extract_embedding(face_img)
                sid, conf = self.system.recognize_face(emb)
                if sid and conf >= self.system.recognition_threshold:
                    name = self.system.students[sid]['name']
                    self.system.log_attendance(sid, name)
                    if sid not in self.seen:
                        now_dt = datetime.now()
                        item = {
                            'student_id': sid,
                            'name': name,
                            'confidence': round(float(conf) * 100.0, 2),
                            'date': now_dt.strftime('%Y-%m-%d'),
                            'time': now_dt.strftime('%H:%M:%S')
                        }
                        try:
                            if self.queue.full():
                                _ = self.queue.get_nowait()
                            self.queue.put_nowait(item)
                            self.seen.add(sid)
                        except Exception:
                            pass
                    self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), name, conf, True)
                else:
                    self.system.draw_face_box_with_name(img, (x1, y1, x2, y2), "Unknown", conf if 'conf' in locals() else 0.0, False)
            self.last_ts = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class AddStudentProcessor(VideoProcessorBase):
    def __init__(self):
        self.system = get_system()
        self.queue = Queue(maxsize=1)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        faces = self.system.detect_faces(img)
        best = None
        area_best = 0
        for f in faces:
            x1, y1, x2, y2 = f['bbox']
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > area_best:
                area_best = area
                best = f
        if best is not None:
            x1, y1, x2, y2 = best['bbox']
            emb = best.get('embedding')
            if emb is None:
                face_img = img[y1:y2, x1:x2]
                if face_img.size > 0:
                    emb = self.system.extract_embedding(face_img)
            if emb is not None:
                try:
                    if self.queue.full():
                        _ = self.queue.get_nowait()
                    self.queue.put_nowait({'embedding': emb})
                except Exception:
                    pass
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Align face and hold", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def page_home():
    st.header("Home")
    st.caption("Choose an action")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add Student", use_container_width=True):
            st.session_state.page = 'add'
            st.rerun()
        if st.button("Delete Student", use_container_width=True):
            st.session_state.page = 'delete'
            st.rerun()
    with c2:
        if st.button("Take Attendance", use_container_width=True):
            st.session_state.page = 'take'
            st.rerun()
        if st.button("List Students", use_container_width=True):
            st.session_state.page = 'list'
            st.rerun()


def page_add():
    system = get_system()
    st.header("Add Student")
    sid = st.text_input("Student ID")
    name = st.text_input("Student Name")
    st.caption("Submit to open the camera, auto-capture one face, and save.")

    # Show success toast/message from last save (after camera stops)
    success_msg = st.session_state.get('add_success_msg')
    if success_msg:
        st.success(success_msg)
        try:
            st.toast(success_msg)
        except Exception:
            pass
        # Clear after showing once
        st.session_state.add_success_msg = None

    # Begin auto flow on submit
    if st.button("Submit & Auto-Capture", type="primary"):
        if not sid or not name:
            st.error("Enter Student ID and Name")
        elif sid in system.students:
            st.warning("Student already exists")
        else:
            st.session_state.add_sid = sid
            st.session_state.add_name = name
            st.session_state.add_active = True
            st.session_state.add_saved = False
            st.rerun()

    add_ctx = None
    if st.session_state.get('add_active', False):
        st.info("Align your face in the frame. Saving automatically...")
        add_ctx = webrtc_streamer(
            key="add_auto",
            video_processor_factory=AddStudentProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Pull the most recent embedding and save once
        if add_ctx and add_ctx.state.playing and add_ctx.video_processor and not st.session_state.get('add_saved', False):
            latest = None
            try:
                while True:
                    latest = add_ctx.video_processor.queue.get_nowait()
            except Empty:
                pass
            if latest is not None:
                emb = latest.get('embedding')
                sid_p = st.session_state.get('add_sid')
                name_p = st.session_state.get('add_name')
                if emb is not None and sid_p and name_p and sid_p not in system.students:
                    ts = datetime.now().isoformat()
                    emb_list = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                    system.students[sid_p] = {'name': name_p, 'registration_date': ts}
                    system.embeddings[sid_p] = {
                        'name': name_p,
                        'embeddings': [{'vector': emb_list, 'timestamp': ts, 'confidence': 1.0, 'uniqueness': 1.0}],
                        'embedding': emb_list,
                        'registration_date': ts,
                        'diversity_score': 1/system.max_embeddings
                    }
                    system.save_data()
                    st.session_state.add_saved = True
                    # Defer message to after rerun so the camera is stopped
                    st.session_state.add_success_msg = f"Saved {name_p} ({sid_p}) successfully."
                    # stop stream and rerun to show popup
                    st.session_state.add_active = False
                    st.rerun()
    else:
        st.info("Fill details and press Submit & Auto-Capture.")


def page_take():
    st.header("Take Attendance")
    system = get_system()
    st.caption("Live camera on left, recognized on right")
    left, right = st.columns([7, 5])
    with left:
        ctx = webrtc_streamer(key="take_attendance", video_processor_factory=TakeAttendanceProcessor, rtc_configuration=RTC_CONFIG, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with right:
        st.subheader("Recognized Students (live)")
        if ctx and ctx.state.playing and ctx.video_processor:
            if 'take_records' not in st.session_state:
                st.session_state.take_records = []
            try:
                while True:
                    item = ctx.video_processor.queue.get_nowait()
                    st.session_state.take_records.append(item)
            except Empty:
                pass
        recs = st.session_state.get('take_records', [])
        # Auto-filter out entries for students that were deleted after recognition
        if recs:
            recs = [r for r in recs if r.get('student_id') in system.students]
            # Keep only the first seen entry per student (unique by ID)
            unique = {}
            for r in recs:
                sid = r.get('student_id')
                if sid and sid not in unique:
                    unique[sid] = r
            recs = list(unique.values())
            st.session_state.take_records = recs
        if not recs:
            st.caption("No one recognized yet.")
        else:
            for i, r in enumerate(recs, 1):
                st.write(f"{i}. {r['name']} (ID: {r['student_id']})\n   Date: {r['date']}   Time: {r['time']}")
        if st.button("Clear Recognitions"):
            st.session_state.take_records = []
            st.rerun()


def page_delete():
    system = get_system()
    st.header("Delete Student")
    if not system.students:
        st.info("No students registered")
        return
    options = [f"{data['name']} (ID: {sid})" for sid, data in system.students.items()]
    choice = st.selectbox("Select", options)
    if st.button("Delete"):
        sid = choice.split("(ID:")[-1].strip().rstrip(")")
        if sid in system.students:
            del system.students[sid]
        if sid in system.embeddings:
            del system.embeddings[sid]
        system.save_data()
        st.success("Deleted")


def page_list():
    system = get_system()
    st.header("List Students")
    if not system.students:
        st.info("No students registered")
        return
    st.write(f"Total: {len(system.students)}")
    for i, (sid, data) in enumerate(system.students.items(), 1):
        st.write(f"{i}. {data['name']} • ID: {sid} • Registered: {data.get('registration_date', 'N/A')}")


def main():
    st.set_page_config(page_title="Face Attendance", layout="wide")
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()
    page = st.session_state.page
    if page == 'home':
        page_home()
    elif page == 'add':
        page_add()
    elif page == 'take':
        page_take()
    elif page == 'delete':
        page_delete()
    elif page == 'list':
        page_list()


if __name__ == "__main__":
    main()


