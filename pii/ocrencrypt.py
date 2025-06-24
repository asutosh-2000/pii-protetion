
import cv2
import re
import pytesseract
import mysql.connector
from mysql.connector import Error
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import random
import time
import logging
import sys
import os
import tempfile

# Database connection parameters
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'pii'
}

# Function to connect to MySQL database


def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        sys.exit(1)

# Preprocess the image to enhance text clarity (grayscale, thresholding)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image file not found or unable to load: {image_path}")

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make text clearer
    _, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

    return thresh_img

# Extract specific sensitive information from the extracted text
def extract_information(text):
    info = {}

    # Aadhaar number extraction: Handling spaces and other separators
    aadhaar_match = re.search(r'\b\d{4}[\s\-_]*\d{4}[\s\-_]*\d{4}\b', text)
    if aadhaar_match:
        aadhaar_number = re.sub(r'[\s\-_]', '', aadhaar_match.group())
        info['Aadhaar'] = aadhaar_number
    else:
        print("Aadhaar number not found")

    # Clean text by removing unnecessary characters and newlines
    clean_text = ' '.join(line.strip() for line in text.split('\n'))

    # Extract Name - Adjusted to handle names spanning multiple lines
    name_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', clean_text)
    if name_match:
        # Extract name with multiple parts
        name = name_match.group(0).strip()
        info['Name'] = name
    else:
        print("Name not found")

    # Extract Date of Birth (DOB in format DD/MM/YYYY)
    dob = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text)
    if dob:
        info['DOB'] = dob.group()

    # Extract Gender
    gender = re.search(r'(MALE|FEMALE|पुरुष|महिला)', text, re.IGNORECASE)
    if gender:
        info['Gender'] = gender.group()

    # Extract Address up to 6-digit pincode
    address = re.search(r'Address:\s*(.*?)(\d{6})', text, re.DOTALL)
    if address:
        full_address = address.group(1).strip().replace('\n', ' ')
        pincode = address.group(2).strip()
        info['Address'] = f"{full_address}, {pincode}"

    # Extract Phone Number (10-digit)
    phone_number = re.search(r'\b\d{10}\b', text)
    if phone_number:
        info['Phone'] = phone_number.group()

    return info


# Extract text from the image using PyTesseract
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image, lang='eng+ori')
    print(f"Extracted Text:\n{text}")
    return text


def blur_faces(image):
    # Load the face detection cascade file
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Detected {len(faces)} face(s).")

    # Apply Gaussian blur to each detected face
    for (x, y, w, h) in faces:
        # Get the face area
        face_area = image[y:y + h, x:x + w]

        # Apply Gaussian blur to the face area
        blurred_face = cv2.GaussianBlur(face_area, (99, 99), 30)

        # Replace the original face with the blurred face
        image[y:y + h, x:x + w] = blurred_face

    return image


def mask_text_in_image(image, target_text):
    # Convert the image to grayscale (optional, improves OCR accuracy)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR to get all the text data and their bounding boxes
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        # Remove extra spaces for accurate comparison
        detected_text = d['text'][i].strip()
        # if detected_text:
        #     print(f"Detected text: '{detected_text}' at position ({d['left'][i]}, {d['top'][i]}) with width {d['width'][i]} and height {d['height'][i]}")

        # Check if the detected text matches the Aadhaar number or part of it
        if detected_text == target_text:
            # Extract the bounding box coordinates
            (x, y, w, h) = (d['left'][i], d['top']
                            [i], d['width'][i], d['height'][i])

            # Draw a rectangle (mask) over the detected text
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

            # Print the coordinates for debugging
            print(f"Masked Aadhaar number '{target_text}' at ({x}, {y}, {w}, {h})")

    return image


def mask_sensitive_data(image, sensitive_info):
    # Extract Aadhaar number
    if 'Aadhaar' in sensitive_info:
        aadhaar_number = sensitive_info['Aadhaar']
        # Mask first 8 digits (first 4, then next 4, leave last 4)
        parts_to_mask = [aadhaar_number[:4], aadhaar_number[4:8]]
        for part in parts_to_mask:
            image = mask_text_in_image(image, part)

    # Extract and mask address
    if 'Address' in sensitive_info:
        address = sensitive_info['Address']
        address_words = address.split()
        for word in address_words:
            image = mask_text_in_image(image, word)

    return image


def read_image_as_binary(image_path):
    try:
        print(f"Reading image from path: {image_path}")  # Debugging
        with open(image_path, 'rb') as file:  # 'rb' to read in binary mode
            binary_data = file.read()
            print(f"Image data read successfully, size: {len(binary_data)} bytes")  # Debugging: Print size
            return binary_data
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None


# Function to pad plaintext to a multiple of AES block size
def pad(data):
    return data + b"\0" * (AES.block_size - len(data) % AES.block_size)

# Function to encrypt the data from the database


# Function to encrypt the data from the database
def encrypt_data(record_id):
    key = get_random_bytes(32)  # Generate a random 256-bit AES key
    iv = get_random_bytes(AES.block_size)  # Generate a random IV

    connection = get_db_connection()
    cursor = connection.cursor()

    # Fetch the row with the specified id from the 'encrypt_input' column
    cursor.execute("SELECT * FROM encryptdecrypt WHERE id = %s", (record_id,))
    row = cursor.fetchone()

    if not row:
        print(f"No data found for id {record_id}.")
        cursor.close()
        connection.close()
        return

    # Ensure encrypt_input is not None before encoding
    plaintext = row[1].encode()
    padded_data = pad(plaintext)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(padded_data)

    # Store the encrypted data, key, and IV in the database
    cursor.execute("""
        UPDATE encryptdecrypt 
        SET encrypt_output = %s, hax_key = %s, encrypt_input = NULL
        WHERE id = %s
    """, (ciphertext, key + iv, record_id))

    connection.commit()

    # Debugging message
    print(f"encrypted at id {record_id}")

    cursor.close()
    connection.close()
    print(f"Encryption completed for record {record_id}.")


# Save extracted information to MySQL database
def save_information_to_database(info, record_id, manager_id, image_path):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        encrypted_input = str(info)

        # Read image as binary data
        image_data = read_image_as_binary(image_path)
        if not image_data:
            raise ValueError("Image data is empty or not read properly.")

        sql_insert_query = """ 
        INSERT INTO encryptdecrypt (id, encrypt_input, encrypt_output, hax_key, decrypt_output, image, manager_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s) """

        insert_tuple = (record_id, encrypted_input, '',
                        '', '', image_data, manager_id)
        cursor.execute(sql_insert_query, insert_tuple)
        connection.commit()

        print(f"Record {record_id} inserted successfully.")

    except Error as e:
        print(f"Database error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def unpad(data):
    return data.rstrip(b"\0")

def decrypt_data(record_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT encrypt_output, hax_key FROM encryptdecrypt WHERE id = %s", (record_id,))
        row = cursor.fetchone()

        if row is None:
            print(f"No data found for record ID {record_id}.")
            return None  # Return None if no data is found

        encrypted_data, key_iv = row
        if encrypted_data is None or key_iv is None:
            print("No encrypted data or key found.")
            return None  # Return None if encrypted data or key is missing

        key = key_iv[:32]
        iv = key_iv[32:]

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(encrypted_data))
        decrypted_text = decrypted_data.decode('utf-8')

        # Update the decrypted output in the database
        cursor.execute("""
            UPDATE encryptdecrypt 
            SET decrypt_output = %s
            WHERE id = %s
        """, (decrypted_text, record_id))
        connection.commit()

        print(f"Decryption successful for record ID {record_id and decrypted_text}.")
        return decrypted_text  # Return the decrypted text

    except (Error, UnicodeDecodeError, ValueError) as e:
        print(f"Decryption error: {e}")
        return None  # Return None if any error occurs

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

            
            
# Main function to process the Aadhaar image, extract data, save, and encrypt
def main(image_path, manager_id):
    record_id = random.randint(1, 10000)  # Generate a random record ID
    print(f"Using record ID: {record_id}")

    # Preprocess the image and extract text
    preprocessed_image = preprocess_image(image_path)
    extracted_text = extract_text_from_image(preprocessed_image)

    # Extract information from the text
    extracted_info = extract_information(extracted_text)
    original_image = cv2.imread(image_path)

    # Blur faces in the original image
    image_with_blurred_faces = blur_faces(original_image.copy())

    # Mask sensitive data in the same image
    final_image = mask_sensitive_data(image_with_blurred_faces, extracted_info)
    print(extracted_info)
    # Save the final masked image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, final_image)
        print(f"Temporary masked image file saved to: {temp_file_path}")

    try:
        # Save extracted information to the database
        save_information_to_database(
            extracted_info, record_id, manager_id, temp_file_path)

        # Encrypt the data in the database
        encrypt_data(record_id)

    finally:
        # Delete the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary masked file {temp_file_path} deleted.")


def tempcheck(image_path):
    # Preprocess the image and extract text
    preprocessed_image = preprocess_image(image_path)
    extracted_text = extract_text_from_image(preprocessed_image)

    # Extract information from the text
    sensitive_info = extract_information(extracted_text)

    # Check if sensitive_info is empty
    if not sensitive_info:
        sensitive_info = {"Message": "No sensitive information detected."}

    # Return the extracted information (even if it's empty)
    return sensitive_info


# Example usage
if __name__ == "_main_":
    image_path = 'temp.jpg'  # Update with your image path
    manager_id = input("Enter Manager ID: ")
    main(image_path, manager_id)

