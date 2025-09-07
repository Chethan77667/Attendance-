import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import DeepFaceAttendance
import json

class AttendanceGUI:
    def __init__(self, master=None):
    
        self.master = master or tk.Tk()
        self.master.title("Face Recognition Attendance System")
        self.master.geometry("400x400")
        self.master.resizable(False, False)

        self.system = DeepFaceAttendance()

        # Title label
        title = tk.Label(self.master, text="Face Recognition Attendance System", font=("Arial", 16, "bold"), pady=20)
        title.pack()

        # Button frame
        btn_frame = tk.Frame(self.master)
        btn_frame.pack(pady=30)

        # Buttons
        btn_add = tk.Button(btn_frame, text="Add Student", width=25, height=2, command=self.add_student)
        btn_add.grid(row=0, column=0, pady=5)

        btn_attend = tk.Button(btn_frame, text="Take Attendance", width=25, height=2, command=self.take_attendance)
        btn_attend.grid(row=1, column=0, pady=5)

        btn_delete = tk.Button(btn_frame, text="Delete Student", width=25, height=2, command=self.delete_student)
        btn_delete.grid(row=2, column=0, pady=5)

        btn_list = tk.Button(btn_frame, text="List Students", width=25, height=2, command=self.list_students)
        btn_list.grid(row=3, column=0, pady=5)

        btn_threshold = tk.Button(btn_frame, text="Set Threshold", width=25, height=2, command=self.set_threshold)
        btn_threshold.grid(row=4, column=0, pady=5)

        btn_exit = tk.Button(btn_frame, text="Exit", width=25, height=2, command=self.exit)
        btn_exit.grid(row=5, column=0, pady=5)

    def add_student(self):
        # Use a form dialog to get student details
        form = tk.Toplevel(self.master)
        form.title("Add Student")
        form.geometry("300x200")
        form.resizable(False, False)
        tk.Label(form, text="Student ID:").pack(pady=5)
        entry_id = tk.Entry(form)
        entry_id.pack(pady=5)
        tk.Label(form, text="Student Name:").pack(pady=5)
        entry_name = tk.Entry(form)
        entry_name.pack(pady=5)
        status_label = tk.Label(form, text="", fg="red")
        status_label.pack(pady=5)
        def submit():
            student_id = entry_id.get().strip()
            name = entry_name.get().strip()
            if not student_id or not name:
                status_label.config(text="Both fields required.")
                return
            if student_id in self.system.students:
                status_label.config(text="Student already exists.")
                return
            form.destroy()
            # Call backend with GUI-based input
            self._add_student_backend(student_id, name)
        tk.Button(form, text="Submit", command=submit).pack(pady=10)
        form.transient(self.master)
        form.grab_set()
        self.master.wait_window(form)

    def _add_student_backend(self, student_id, name):
        # This runs the camera and face capture as in main.py, but with GUI feedback
        def run_capture():
            import cv2
            from datetime import datetime
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Camera not found.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            captured_embedding = None
            capture_attempts = 0
            max_attempts = 50
            while capture_attempts < max_attempts and captured_embedding is None:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                faces = self.system.detect_faces(frame)
                cv2.putText(frame, f"Looking for face... ({capture_attempts}/{max_attempts})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Position your face in the camera", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                for face in faces:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, "Detecting...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    if face['embedding'] is not None:
                        captured_embedding = face['embedding']
                        break
                    else:
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            embedding = self.system.extract_embedding(face_img)
                            if embedding is not None:
                                captured_embedding = embedding
                                break
                cv2.imshow('Face Capture', frame)
                key = cv2.waitKey(100) & 0xFF
                if key == 27:
                    break
                capture_attempts += 1
            cap.release()
            cv2.destroyAllWindows()
            if captured_embedding is not None:
                self.system.students[student_id] = {
                    'name': name,
                    'registration_date': datetime.now().isoformat()
                }
                embedding_entry = {
                    'vector': captured_embedding.tolist(),
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 1.0,
                    'uniqueness': 1.0
                }
                self.system.embeddings[student_id] = {
                    'name': name,
                    'embeddings': [embedding_entry],
                    'embedding': captured_embedding.tolist(),
                    'registration_date': datetime.now().isoformat(),
                    'diversity_score': 1/self.system.max_embeddings
                }
                self.system.save_data()
                messagebox.showinfo("Success", f"Student {name} registered successfully!")
            else:
                messagebox.showerror("Error", "Failed to capture face - please try again.")
        threading.Thread(target=run_capture).start()

    def take_attendance(self):
        # Open a window to show recognized students
        attendance_window = tk.Toplevel(self.master)
        attendance_window.title("Attendance - Recognized Students")
        attendance_window.geometry("350x400")
        attendance_window.resizable(False, False)
        tk.Label(attendance_window, text="Recognized Students", font=("Arial", 14, "bold"), pady=10).pack()
        listbox = tk.Listbox(attendance_window, width=40, height=15)
        listbox.pack(pady=10)
        status_label = tk.Label(attendance_window, text="Camera running... Press ESC in camera window to stop.")
        status_label.pack(pady=5)
        recognized_students = set()
        def safe_insert(text):
            self.master.after(0, lambda: listbox.insert(tk.END, text))
        def safe_status(text):
            self.master.after(0, lambda: status_label.config(text=text))
        def run_attendance():
            import cv2
            from datetime import datetime
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Camera not found.")
                self.master.after(0, attendance_window.destroy)
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            frame_count = 0
            start_time = time.time()
            camera_start_time = start_time
            last_recognition_time = 0
            recognition_interval = 0.3
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame_count += 1
                current_time = time.time()
                faces = self.system.detect_faces(frame)
                if current_time - last_recognition_time >= recognition_interval:
                    for face in faces:
                        bbox = face['bbox']
                        x1, y1, x2, y2 = bbox
                        face_embedding = face['embedding']
                        if face_embedding is None:
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size > 0:
                                face_embedding = self.system.extract_embedding(face_img)
                        student_id, confidence = self.system.recognize_face(face_embedding)
                        if student_id and confidence >= self.system.recognition_threshold:
                            name = self.system.students[student_id]['name']
                            if student_id not in recognized_students:
                                recognized_students.add(student_id)
                                safe_insert(f"{name} (ID: {student_id}) - {confidence*100:.2f}%")
                            self.system.log_attendance(student_id, name)
                            self.system.draw_face_box_with_name(frame, bbox, name, confidence, True)
                        else:
                            self.system.draw_face_box_with_name(frame, bbox, "Unknown", confidence, False)
                    last_recognition_time = current_time
                else:
                    for face in faces:
                        bbox = face['bbox']
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, "Tracking...", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"FPS: {frame_count/(current_time-start_time):.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Recognized: {len(recognized_students)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Auto-Learning Active", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Time: {current_time-camera_start_time:.2f}s", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press ESC to stop", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('DeepFace Attendance', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
            safe_status(f"Attendance completed! {len(recognized_students)} students recognized.")
        threading.Thread(target=run_attendance).start()

    def delete_student(self):
        students = list(self.system.students.items())
        if not students:
            messagebox.showinfo("Delete Student", "No students registered.")
            return
        options = [f"{data['name']} (ID: {student_id})" for student_id, data in students]
        choice = simpledialog.askinteger("Delete Student", "Enter student number to delete:\n" + '\n'.join(f"{i+1}. {opt}" for i, opt in enumerate(options)))
        if choice is None:
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(students):
                student_id, student_data = students[idx]
                confirm = messagebox.askyesno("Confirm Delete", f"Delete {student_data['name']}?")
                if confirm:
                    del self.system.students[student_id]
                    if student_id in self.system.embeddings:
                        del self.system.embeddings[student_id]
                    self.system.save_data()
                    messagebox.showinfo("Delete Student", f"{student_data['name']} deleted.")
            else:
                messagebox.showerror("Error", "Invalid choice.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def list_students(self):
        # Always reload from disk to reflect changes from other UIs (e.g., Streamlit)
        try:
            self.system.load_data()
        except Exception:
            pass

        students = self.system.students
        # Fallback: if in-memory is empty, read the JSON directly
        if not students:
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                data_file = os.path.join(base_dir, 'database', 'students_deepface.json')
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        students = json.load(f)
            except Exception:
                pass
        if not students:
            messagebox.showinfo("List Students", "No students registered.")
            return
        info = f"Total students: {len(students)}\n\n"
        for i, (student_id, data) in enumerate(students.items(), 1):
            info += f"{i}. {data['name']}\n   ID: {student_id}\n   Registered: {data.get('registration_date', 'N/A')}\n\n"
        # Show in a scrollable window
        list_win = tk.Toplevel(self.master)
        list_win.title("Registered Students")
        list_win.geometry("350x400")
        text = tk.Text(list_win, wrap=tk.WORD, width=40, height=20)
        text.insert(tk.END, info)
        text.config(state=tk.DISABLED)
        text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Button(list_win, text="Close", command=list_win.destroy).pack(pady=5)

    def set_threshold(self):
        current = self.system.recognition_threshold
        new_val = simpledialog.askfloat("Set Threshold", f"Current threshold: {current:.2f}\nEnter new threshold (0.1-0.9):", minvalue=0.1, maxvalue=0.9)
        if new_val is not None:
            if 0.1 <= new_val <= 0.9:
                self.system.recognition_threshold = new_val
                messagebox.showinfo("Set Threshold", f"Threshold set to {new_val:.2f}")
            else:
                messagebox.showerror("Error", "Value must be between 0.1 and 0.9.")

    def exit(self):
        self.master.quit()

if __name__ == "__main__":
    gui = AttendanceGUI()
    gui.master.mainloop() 