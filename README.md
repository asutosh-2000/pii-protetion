# PII Protect System

---

## Idea / Approach Details
The **PII Protect System** is designed as a web application with the following functionality:
- **Image Upload**: Users can upload images containing PII.
- **OCR Extraction**: Optical Character Recognition (OCR) is used to extract details from the document.
- **PII Detection**: The system checks for sensitive information.
- **User Control**: Based on user input, the system either displays the data or verifies the manager's ID before sending it securely.
- **Data Encryption**: All extracted text is encrypted using AES, and the masked document is stored as a secure blob.
- **Manager Access**: Managers can view and manage user data after logging in with their ID.

---

## Technical Stack
### Frontend
![HTML](https://img.shields.io/badge/-HTML5-E34F26?logo=html5&logoColor=fff&style=flat-square)
![CSS](https://img.shields.io/badge/-CSS3-1572B6?logo=css3&logoColor=fff&style=flat-square)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?logo=javascript&logoColor=000&style=flat-square)

### Backend
![Django](https://img.shields.io/badge/-Django-092E20?logo=django&logoColor=fff&style=flat-square)
![Node.js](https://img.shields.io/badge/-Node.js-339933?logo=node.js&logoColor=fff&style=flat-square)
![MySQL](https://img.shields.io/badge/-MySQL-4479A1?logo=mysql&logoColor=fff&style=flat-square)
- **Encryption**: AES for securing sensitive data
- **Storage**: Blob storage for encrypted document storage
- **OCR**: Tesseract for text extraction

---
![Workflow](https://github.com/user-attachments/assets/778976b0-4234-4965-be84-600fc2553c17)
---

## Feasibility and Viability
- **High Demand**: Increasing digital records handling PII like Aadhaar cards make this system relevant.
- **Technological Infrastructure**: Web-based systems are feasible due to India’s robust internet infrastructure and adoption of cloud services.
- **Regulatory Compliance**: Aligns with data protection laws like IT Act, DPDP Bill 2023, etc.
- **Challenges**: Ensuring user awareness, managing managerial misuse, and addressing potential data breaches.

---

## Impact and Benefits
- **Enhanced Privacy Protection**: Reduces risks of identity theft and unauthorized data access.
- **Increased Trust**: Strong encryption and access control build user confidence.
- **Educational Value**: Promotes responsible data handling by users and managers.
- **Compliance**: Ensures adherence to India’s data protection laws.

---

## Potential Challenges
1. **Regulatory Compliance**: Complex rules around the DPDP Bill.
2. **User Awareness**: Low user knowledge of data masking and security.
3. **Managerial Misuse**: Risk of sensitive data misuse by managers.
4. **Cybersecurity Threats**: Despite encryption, cyberattacks remain a risk.

---


