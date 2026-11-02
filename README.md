A real-time computer vision application that allows users to draw Sinhala characters in the air using hand gestures and translates them into digital text using a custom Convolutional Neural Network (CNN) trained on a small, independently collected dataset.

🚀 Features

Real-Time Hand Tracking: Uses Google MediaPipe to track index finger coordinates with high precision.

Mathematical Smoothing: Implements an Exponential Moving Average (EMA) filter to absorb natural hand jitter, creating smooth, calligraphy-like ink strokes.

Custom Small Dataset Training: Successfully trains on a limited, custom-built dataset of Sinhala characters by utilizing heavy data augmentation (rotation, translation, zoom) and Batch Normalization to prevent overfitting.

Deep CNN Classifier: A custom TensorFlow/Keras model optimized for 64x64 pixel resolution to preserve and recognize the complex internal loops of Sinhala characters (අ, එ, ඉ, උ, ස, ර, ව).

Dynamic Image Preprocessing: Automatically extracts bounding boxes around drawn ink, scales the drawing while maintaining aspect ratio, and normalizes the live webcam input to perfectly match the static training dataset's distribution.

Interactive UI: Features on-screen hover buttons for color selection and keyboard shortcuts for rapid canvas clearing and predictions.

🛠️ Tech Stack

Language: Python 3.12

Computer Vision: OpenCV, MediaPipe

Machine Learning: TensorFlow (Legacy Keras), Scikit-Learn

Data Processing: NumPy, Pillow (PIL)

📂 Project Structure

drawing.py: The main application script handling the webcam feed, hand tracking, UI, and live predictions.

train_model.py: The Deep CNN architecture and training pipeline utilizing data augmentation and Keras Callbacks (EarlyStopping, ReduceLROnPlateau) to maximize accuracy on a small dataset.

dataset/: Contains the small, custom-created 64x64 image dataset used to train the model, sorted into folders by character class.

drawing_model.keras: The compiled, trained model file.

🎮 How to Use

Controls:

Draw: Pinch your index finger and thumb apart (> 40 pixels) to drop ink. Bring them together to pause drawing.

Change Colors: Hover your index finger over the Red, Green, Blue, or White buttons at the top of the screen.

ENTER: Process the canvas and predict the Sinhala character.

'c' Key: Clear the entire canvas.

'q' Key: Quit the application.
