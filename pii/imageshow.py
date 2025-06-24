import mysql.connector
from mysql.connector import Error
import os

def fetch_image_by_id(user_id):
    try:
        # Establish a database connection
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='pii'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch the image based on the given id
            query = "SELECT image FROM encryptdecrypt WHERE id = %s"
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()

            if result:
                image_data = result[0]
                
                # Define the path to save the image
                directory = 'pii/static/fetchimg'
                
                # Ensure the directory exists
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                # Construct the file path
                file_path = os.path.join(directory, f'image_{user_id}.jpg')
                
                # Save the image to the file
                with open(file_path, 'wb') as file:
                    file.write(image_data)
                
                print(f'Image with ID {user_id} has been saved successfully to {file_path}.')
            else:
                print(f'No image found with ID {user_id}.')

    except Error as e:
        print(f'Error: {e}')
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Example call to the function (you can replace '3307' with your specific user_id)
# fetch_image_by_id(user_id)
