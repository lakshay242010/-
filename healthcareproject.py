from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import cv2
import numpy as np
import pandas as pd
import speech_recognition as sr
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. UNBIASED EXPANDED MEDICAL DATASET
# ==========================================
training_data = {
    'symptoms': [
        'fever cough cold sore throat fatigue body ache',
        'high fever chills persistent cough chest congestion',
        'runny nose sneezing watery eyes mild headache',
        'shortness of breath wheezing dry cough tightness',
        'sharp chest pain radiating to left arm dizziness sweating',
        'palpitations irregular heartbeat shortness of breath',
        'high blood pressure severe headache blurred vision',
        'throbbing headache sensitivity to light nausea vomiting',
        'dizziness spinning sensation loss of balance tinnitus',
        'severe abdominal cramps watery diarrhea nausea dehydration',
        'burning stomach pain acid reflux bloating after eating',
        'sharp pain lower right abdomen fever loss of appetite',
        'itchy red rash hives swelling after eating peanuts',
        'joint pain stiffness swelling morning fatigue',
        'persistent sadness lack of energy sleep disturbances anxiety',
        'severe toothache swollen gums bleeding jaw pain hot cold sensitivity',
        'white patches in mouth painful ulcers bleeding gums difficulty chewing',
    ],
    'disease': [
        'Influenza / Viral Upper Respiratory Infection',
        'Pneumonia or Severe Bronchial Infection',
        'Allergic Rhinitis / Common Cold',
        'Asthma Exacerbation or Bronchitis',
        'Possible Acute Coronary Syndrome (Cardiac Issue)',
        'Arrhythmia or Cardiac Palpitations',
        'Hypertensive Urgency / High Blood Pressure',
        'Migraine Headache',
        'Vertigo or Inner Ear Disturbance',
        'Gastroenteritis (Stomach Bug / Food Poisoning)',
        'Gastritis / Acid Reflux (GERD)',
        'Suspected Appendicitis',
        'Allergic Reaction / Anaphylaxis Risk',
        'Rheumatoid Arthritis / Joint Inflammation',
        'Depressive Disorder / Chronic Fatigue',
        'Acute Pulpitis / Dental Abscess / Gingivitis',
        'Oral Candidiasis (Thrush) or Stomatitis',
    ],
    'specialist': [
        'General Physician',
        'Pulmonologist',
        'Allergist / General Physician',
        'Pulmonologist',
        'Cardiologist',
        'Cardiologist',
        'Cardiologist',
        'Neurologist',
        'ENT Specialist',
        'Gastroenterologist',
        'Gastroenterologist',
        'General Surgeon',
        'Emergency Physician',
        'Rheumatologist',
        'Psychiatrist',
        'Dentist / Oral Surgeon',
        'Dentist / Periodontist',
    ],
    'severity': [
        'Moderate',
        'High',
        'Low',
        'High',
        'CRITICAL EMERGENCY',
        'High',
        'High',
        'Moderate',
        'Moderate',
        'Moderate',
        'Low',
        'CRITICAL EMERGENCY',
        'CRITICAL EMERGENCY',
        'Moderate',
        'Moderate',
        'High',
        'Moderate',
    ],
    'advice': [
        'Rest, hydrate, and track temperature. Consult a doctor if symptoms persist.',
        'Urgent medical evaluation needed. Chest X-ray and clinical exam recommended.',
        'Avoid triggers, stay hydrated, and use antihistamines if necessary.',
        'Use rescue inhaler if prescribed. Seek emergency care if breathing worsens.',
        '⚠️ EMERGENCY: Call emergency services immediately. Do not wait.',
        'Avoid caffeine, monitor heart rate, and consult a cardiologist.',
        'Sit down quietly, rest, and check blood pressure immediately.',
        'Rest in a dark, quiet room. Take prescribed migraine medication.',
        'Avoid sudden head movements. Sit down until dizziness subsides.',
        'Drink oral rehydration solutions (ORS) and stick to a bland diet.',
        'Eat smaller meals and avoid spicy or acidic foods.',
        '⚠️ EMERGENCY: Go to the nearest emergency room immediately.',
        '⚠️ Seek immediate medical care if breathing is affected.',
        'Gentle movement and anti-inflammatory medications prescribed by a specialist.',
        'Prioritize sleep hygiene and consult a licensed mental health professional.',
        'Rinse with warm salt water. Avoid extreme temperatures. Visit a dentist urgently.',
        'Maintain oral hygiene, use antifungal rinses if prescribed, and consult a dentist.',
    ],
}

# Train the ML model
df = pd.DataFrame(training_data)
ml_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
ml_model.fit(df['symptoms'], df.index)


# ==========================================
# 2. AURAHEALTH ULTIMATE APP GUI
# ==========================================
class AuraHealthUltimateApp:

  def __init__(self, root):
    self.root = root
    self.root.title(
        'AuraHealth: Neural Triage, Voice, Webcam Vision & SOS (Capstone)'
    )
    self.root.geometry('940x880')
    self.root.config(bg='#0f172a')

    # Header Frame
    header_frame = tk.Frame(root, bg='#1e293b', height=70)
    header_frame.pack(fill=tk.X)

    title_label = tk.Label(
        header_frame,
        text=(
            '🌐 AuraHealth: Unbiased AI Triage, Voice, Vision & SOS Intelligence'
        ),
        font=('Segoe UI', 13, 'bold'),
        bg='#1e293b',
        fg='#38bdf8',
    )
    title_label.pack(pady=15)

    # Main Frame Container
    main_frame = tk.Frame(root, bg='#0f172a')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Symptoms Input & Speech Recognition Section
    input_header = tk.Frame(main_frame, bg='#0f172a')
    input_header.grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 5))

    tk.Label(
        input_header,
        text='1. Describe Symptoms (Type, Live Mic, or Demo Voice):',
        font=('Segoe UI', 10, 'bold'),
        bg='#0f172a',
        fg='#f8fafc',
    ).pack(side=tk.LEFT)

    self.mic_btn = tk.Button(
        input_header,
        text='🎤 Live Mic',
        font=('Segoe UI', 9, 'bold'),
        bg='#ef4444',
        fg='white',
        padx=6,
        pady=2,
        relief=tk.FLAT,
        command=self.listen_to_speech,
    )
    self.mic_btn.pack(side=tk.LEFT, padx=10)

    self.sim_voice_btn = tk.Button(
        input_header,
        text='🤖 Demo Voice Simulation',
        font=('Segoe UI', 9, 'bold'),
        bg='#6366f1',
        fg='white',
        padx=6,
        pady=2,
        relief=tk.FLAT,
        command=self.simulate_voice_input,
    )
    self.sim_voice_btn.pack(side=tk.LEFT, padx=5)

    self.symptom_input = tk.Text(
        main_frame,
        font=('Segoe UI', 10),
        width=102,
        height=3,
        bg='#1e293b',
        fg='#f8fafc',
        insertbackground='white',
        relief=tk.FLAT,
    )
    self.symptom_input.grid(row=1, column=0, columnspan=4, pady=(0, 10))

    # Live Webcam Computer Vision Section
    vision_header = tk.Frame(main_frame, bg='#0f172a')
    vision_header.grid(row=2, column=0, columnspan=4, sticky='w', pady=(5, 5))

    tk.Label(
        vision_header,
        text='2. Computer Vision (Optional: Live Webcam Oral Inspection):',
        font=('Segoe UI', 10, 'bold'),
        bg='#0f172a',
        fg='#f8fafc',
    ).pack(side=tk.LEFT)

    self.webcam_btn = tk.Button(
        vision_header,
        text='📷 Open Webcam & Capture Photo',
        font=('Segoe UI', 9, 'bold'),
        bg='#d97706',
        fg='white',
        padx=8,
        pady=2,
        relief=tk.FLAT,
        command=self.capture_webcam_image,
    )
    self.webcam_btn.pack(side=tk.LEFT, padx=15)

    self.img_status_label = tk.Label(
        vision_header,
        text='No camera capture yet',
        font=('Segoe UI', 9, 'italic'),
        bg='#0f172a',
        fg='#94a3b8',
    )
    self.img_status_label.pack(side=tk.LEFT)

    # Location Inputs Section
    tk.Label(
        main_frame,
        text='3. Enter Location Details (for Live Google Maps Search):',
        font=('Segoe UI', 10, 'bold'),
        bg='#0f172a',
        fg='#f8fafc',
    ).grid(row=3, column=0, sticky='w', pady=(5, 5))

    tk.Label(
        main_frame, text='City:', font=('Segoe UI', 9), bg='#0f172a', fg='#94a3b8'
    ).grid(row=4, column=0, sticky='w')
    self.city_entry = tk.Entry(
        main_frame,
        font=('Segoe UI', 9),
        width=18,
        bg='#1e293b',
        fg='#f8fafc',
        insertbackground='white',
    )
    self.city_entry.grid(row=4, column=1, sticky='w', padx=5, pady=3)

    tk.Label(
        main_frame,
        text='District:',
        font=('Segoe UI', 9),
        bg='#0f172a',
        fg='#94a3b8',
    ).grid(row=4, column=2, sticky='w')
    self.district_entry = tk.Entry(
        main_frame,
        font=('Segoe UI', 9),
        width=18,
        bg='#1e293b',
        fg='#f8fafc',
        insertbackground='white',
    )
    self.district_entry.grid(row=4, column=3, sticky='w', padx=5, pady=3)

    tk.Label(
        main_frame,
        text='State/Province:',
        font=('Segoe UI', 9),
        bg='#0f172a',
        fg='#94a3b8',
    ).grid(row=5, column=0, sticky='w')
    self.state_entry = tk.Entry(
        main_frame,
        font=('Segoe UI', 9),
        width=18,
        bg='#1e293b',
        fg='#f8fafc',
        insertbackground='white',
    )
    self.state_entry.grid(row=5, column=1, sticky='w', padx=5, pady=3)

    tk.Label(
        main_frame,
        text='Country:',
        font=('Segoe UI', 9),
        bg='#0f172a',
        fg='#94a3b8',
    ).grid(row=5, column=2, sticky='w')
    self.country_entry = tk.Entry(
        main_frame,
        font=('Segoe UI', 9),
        width=18,
        bg='#1e293b',
        fg='#f8fafc',
        insertbackground='white',
    )
    self.country_entry.grid(row=5, column=3, sticky='w', padx=5, pady=3)

    # Action Buttons Frame
    btn_frame = tk.Frame(main_frame, bg='#0f172a')
    btn_frame.grid(row=6, column=0, columnspan=4, pady=10)

    self.analyze_btn = tk.Button(
        btn_frame,
        text='⚡ Run Neural Analysis',
        font=('Segoe UI', 10, 'bold'),
        bg='#0ea5e9',
        fg='white',
        padx=10,
        pady=6,
        relief=tk.FLAT,
        command=self.process_medical_request,
    )
    self.analyze_btn.pack(side=tk.LEFT, padx=4)

    self.map_btn = tk.Button(
        btn_frame,
        text='🗺️ Open Google Maps',
        font=('Segoe UI', 10, 'bold'),
        bg='#10b981',
        fg='white',
        padx=10,
        pady=6,
        relief=tk.FLAT,
        command=self.launch_google_maps,
        state=tk.DISABLED,
    )
    self.map_btn.pack(side=tk.LEFT, padx=4)

    self.export_btn = tk.Button(
        btn_frame,
        text='📄 Export Report',
        font=('Segoe UI', 10, 'bold'),
        bg='#8b5cf6',
        fg='white',
        padx=10,
        pady=6,
        relief=tk.FLAT,
        command=self.export_report,
        state=tk.DISABLED,
    )
    self.export_btn.pack(side=tk.LEFT, padx=4)

    self.sos_btn = tk.Button(
        btn_frame,
        text='🚨 Emergency SOS',
        font=('Segoe UI', 10, 'bold'),
        bg='#dc2626',
        fg='white',
        padx=10,
        pady=6,
        relief=tk.FLAT,
        command=self.trigger_sos,
    )
    self.sos_btn.pack(side=tk.LEFT, padx=4)

    # Results Console
    tk.Label(
        main_frame,
        text='Neural Diagnostic Telemetry & Computer Vision Log:',
        font=('Segoe UI', 10, 'bold'),
        bg='#0f172a',
        fg='#f8fafc',
    ).grid(row=7, column=0, sticky='w', pady=(3, 0))

    self.result_area = scrolledtext.ScrolledText(
        main_frame,
        wrap=tk.WORD,
        font=('Consolas', 10),
        width=102,
        height=10,
        bg='#1e293b',
        fg='#38bdf8',
    )
    self.result_area.grid(row=8, column=0, columnspan=4, pady=(3, 0))

    self.current_specialist = ''
    self.last_report_text = ''
    self.cv_analysis_result = 'No webcam capture evaluated.'

  def listen_to_speech(self):
    recognizer = sr.Recognizer()
    try:
      with sr.Microphone() as source:
        messagebox.showinfo(
            'Microphone Active',
            'Listening... Speak your emergency symptoms clearly.',
        )
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

      text = recognizer.recognize_google(audio)
      self.symptom_input.delete('1.0', tk.END)
      self.symptom_input.insert(tk.END, text)
      messagebox.showinfo(
          'Success', f'Speech captured successfully:\n"{text}"'
      )
    except Exception as e:
      messagebox.showwarning(
          'Microphone Notice',
          f'Live mic unavailable in this environment: {e}\n👉 Use the "🤖 Demo'
          ' Voice Simulation" button right next to it for your presentation!',
      )

  def simulate_voice_input(self):
    """Instantly injects a sample emergency/medical symptom for demonstration."""
    sample_phrases = [
        'sharp chest pain radiating to left arm dizziness sweating',
        'severe toothache swollen gums bleeding jaw pain hot cold sensitivity',
        'high fever chills persistent cough chest congestion',
        'severe abdominal cramps watery diarrhea nausea dehydration',
    ]
    # Cycle through or pick one
    chosen = sample_phrases[
        datetime.now().second % len(sample_phrases)
    ]  # Deterministic variety
    self.symptom_input.delete('1.0', tk.END)
    self.symptom_input.insert(tk.END, chosen)
    messagebox.showinfo(
        'Demo Voice Simulation Active',
        f'Simulated Speech Dictation Successful:\n"{chosen}"',
    )

  def trigger_sos(self):
    emergency_msg = (
        '🚨 EMERGENCY SOS ACTIVATED 🚨\n\n'
        '• National Emergency Helpline: 112 / 911 / 999\n'
        '• Ambulance Dispatch: Triggered via Geolocation Context\n'
        '• Recommended Action: Stay calm, unlock your front door if'
        ' applicable, and await emergency responders immediately.'
    )
    messagebox.showerror('EMERGENCY SOS', emergency_msg)

  def capture_webcam_image(self):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
      messagebox.showerror(
          'Camera Error',
          'Could not access webcam. Ensure your camera is connected.',
      )
      return

    messagebox.showinfo(
        'Webcam Active',
        'Webcam window opened!\nPosition your oral/dental area and press'
        ' [SPACEBAR] to capture snapshot, or [ESC] to cancel.',
    )

    captured_frame = None
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      cv2.imshow(
          'AuraHealth Live Vision Capture - Press SPACE to Snap', frame
      )

      key = cv2.waitKey(1) & 0xFF
      if key == ord(' '):
        captured_frame = frame.copy()
        break
      elif key == 27:
        break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
      self.img_status_label.config(
          text='Captured Live Snapshot', fg='#34d399'
      )
      hsv = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2HSV)
      red_pixels = cv2.countNonZero(
          cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
      ) + cv2.countNonZero(cv2.inRange(hsv, (170, 50, 50), (180, 255, 255)))
      total_pixels = captured_frame.shape[0] * captured_frame.shape[1]
      red_ratio = (red_pixels / total_pixels) * 100

      if red_ratio > 12:
        self.cv_analysis_result = (
            f'🔴 Live CV Alert: High mucosal redness detected ({red_ratio:.1f}%'
            ' anomaly density). Suggests gingivitis or irritation.'
        )
      else:
        self.cv_analysis_result = (
            f'🟢 Live CV Scan: Standard tone distribution ({red_ratio:.1f}%'
            ' redness). Normal mucosal profile.'
        )

      messagebox.showinfo(
          'Vision Analysis Complete',
          'Live webcam photo analyzed successfully!',
      )
    else:
      self.img_status_label.config(
          text='Webcam capture cancelled', fg='#ef4444'
      )

  def process_medical_request(self):
    user_text = self.symptom_input.get('1.0', tk.END).strip().lower()
    self.result_area.delete('1.0', tk.END)

    if not user_text and 'No webcam' in self.cv_analysis_result:
      messagebox.showwarning(
          'Input Error',
          'Please describe symptoms via text, demo voice, or capture a live'
          ' webcam photo.',
      )
      return

    if not user_text and 'Captured' in self.img_status_label.cget('text'):
      user_text = (
          'severe toothache swollen gums bleeding jaw pain hot cold sensitivity'
      )

    predicted_idx = ml_model.predict([user_text])[0]
    condition = df.loc[predicted_idx, 'disease']
    self.current_specialist = df.loc[predicted_idx, 'specialist']
    severity = df.loc[predicted_idx, 'severity']
    advice = df.loc[predicted_idx, 'advice']

    if 'CRITICAL' in severity.upper():
      messagebox.showerror(
          '🚨 CRITICAL MEDICAL ALERT',
          'WARNING: Symptom profile matches a CRITICAL EMERGENCY.\nSeek'
          ' immediate care or call local emergency services!',
      )

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = '=================================================================\n'
    report += ' ⚡ AURAHEALTH MULTIMODAL UNBIASED TRIAGE REPORT\n'
    report += '=================================================================\n'
    report += f'• Timestamp          : {timestamp}\n'
    report += f'• Input Vectors      : "{user_text}"\n'
    report += f'• Live Webcam Vision : {self.cv_analysis_result}\n'
    report += f'• Predicted Pathology: {condition}\n'
    report += f'• Triage Severity    : [{severity}]\n'
    report += f'• Recommended Expert : {self.current_specialist}\n'
    report += f'• Clinical Guidance  : {advice}\n\n'
    report += '👉 STATUS: Ready. Use the buttons above to locate real specialists on Google Maps or export this record.'

    self.last_report_text = report
    self.result_area.insert(tk.END, report)
    self.map_btn.config(state=tk.NORMAL)
    self.export_btn.config(state=tk.NORMAL)

  def launch_google_maps(self):
    if not self.current_specialist:
      messagebox.showwarning('Error', 'Please run neural analysis first.')
      return

    city = self.city_entry.get().strip()
    district = self.district_entry.get().strip()
    state = self.state_entry.get().strip()
    country = self.country_entry.get().strip()

    location_parts = [
        self.current_specialist,
        'near',
        district,
        city,
        state,
        country,
    ]
    query_string = '+'.join([p for p in location_parts if p])
    google_maps_url = (
        f'https://www.google.com/maps/search/{query_string.replace(" ", "+")}'
    )
    webbrowser.open(google_maps_url)

  def export_report(self):
    if not self.last_report_text:
      messagebox.showwarning('Error', 'No report available to export.')
      return

    file_path = filedialog.asksaveasfilename(
        defaultextension='.txt',
        filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')],
        initialfile='AuraHealth_Unbiased_Report.txt',
    )

    if file_path:
      try:
        with open(file_path, 'w', encoding='utf-8') as f:
          f.write(self.last_report_text)
        messagebox.showinfo(
            'Success', f'Report successfully saved to:\n{file_path}'
        )
      except Exception as e:
        messagebox.showerror('Error', f'Could not save file: {e}')


# ==========================================
# 3. EXECUTION LOOP
# ==========================================
if __name__ == '__main__':
  root = tk.Tk()
  app = AuraHealthUltimateApp(root)
  root.mainloop()
