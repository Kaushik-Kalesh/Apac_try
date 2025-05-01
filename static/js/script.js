document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const inputChoice = document.getElementById('inputChoice');
    const uploadSection = document.getElementById('uploadSection');
    const liveSection = document.getElementById('liveSection');
    const textSection = document.getElementById('textSection');
    
    // Upload Image Elements
    const uploadBtn = document.getElementById('uploadBtn');
    const imageUpload = document.getElementById('imageUpload');
    const uploadedImage = document.getElementById('uploadedImage');
    const uploadOutput = document.getElementById('uploadOutput');
    const copyUploadBtn = document.getElementById('copyUploadBtn');
    const uploadType = document.getElementById('uploadType');
    
    // Live Webcam Elements
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');
    const resetBtn = document.getElementById('resetBtn');
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const liveOutput = document.getElementById('liveOutput');
    const copyLiveBtn = document.getElementById('copyLiveBtn');
    const chatbotResponse = document.getElementById('chatbotResponse');
    
    // Text to Sign Elements
    const textInput = document.getElementById('textInput');
    const translateTextBtn = document.getElementById('translateTextBtn');
    const signImages = document.getElementById('signImages');
    
    // Variables for live translation
    let stream = null;
    let lastPredictionTime = 0;
    let previousWord = null;
    let sentence = '';
    let isWebcamRunning = false;
    
    // Event Listeners
    inputChoice.addEventListener('change', toggleInputSection);
    uploadBtn.addEventListener('click', handleImageUpload);
    copyUploadBtn.addEventListener('click', () => copyToClipboard(uploadOutput));
    startWebcamBtn.addEventListener('click', startWebcam);
    stopWebcamBtn.addEventListener('click', stopWebcam);
    resetBtn.addEventListener('click', resetTranslation);
    copyLiveBtn.addEventListener('click', () => copyToClipboard(liveOutput));
    translateTextBtn.addEventListener('click', handleTextToSign);
    
    // Functions
    function toggleInputSection() {
        const choice = inputChoice.value;
        uploadSection.classList.add('d-none');
        liveSection.classList.add('d-none');
        textSection.classList.add('d-none');
        
        if (choice === 'upload') {
            uploadSection.classList.remove('d-none');
        } else if (choice === 'live') {
            liveSection.classList.remove('d-none');
        } else if (choice === 'text') {
            textSection.classList.remove('d-none');
        }
    }

    
    
    function handleImageUpload() {
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('choice', uploadType.value);
        
        fetch('/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadOutput.value = data.error;
                uploadedImage.classList.add('d-none');
            } else {
                uploadedImage.src = data.image_url;
                uploadedImage.classList.remove('d-none');
                uploadOutput.value = data.prediction;
                
                // Check for chatbot response
                checkChatbotResponse(data.prediction);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            uploadOutput.value = 'An error occurred during translation';
        });
    }
    
    function startWebcam() {
        if (isWebcamRunning) return;
        
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(mediaStream) {
                    stream = mediaStream;
                    webcam.srcObject = mediaStream;
                    webcam.classList.remove('d-none');
                    canvas.classList.remove('d-none');
                    startWebcamBtn.classList.add('d-none');
                    stopWebcamBtn.classList.remove('d-none');
                    isWebcamRunning = true;
                    
                    // Start processing frames
                    processFrame();
                })
                .catch(function(error) {
                    console.error('Error accessing webcam:', error);
                    alert('Could not access the webcam. Please ensure you have granted permission.');
                });
        } else {
            alert('getUserMedia is not supported in your browser');
        }
    }
    
    function stopWebcam() {
        if (!isWebcamRunning) return;
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
            webcam.classList.add('d-none');
            canvas.classList.add('d-none');
            startWebcamBtn.classList.remove('d-none');
            stopWebcamBtn.classList.add('d-none');
            isWebcamRunning = false;
        }
    }
    
    function resetTranslation() {
        sentence = '';
        liveOutput.value = '';
        chatbotResponse.classList.add('d-none');
        previousWord = null;
    }
    
    function processFrame() {
        if (!isWebcamRunning) return;
        
        const context = canvas.getContext('2d');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        
        const imageData = canvas.toDataURL('image/jpeg');
        
        fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.has_hands) {
                const currentTime = Date.now();
                
                if (currentTime - lastPredictionTime >= 4000) { // 4 seconds
                    const predictedLabel = data.prediction;
                    
                    if (predictedLabel !== previousWord) {
                        if (['How', 'how', 'What', 'what', 'where', 'Where'].includes(previousWord) && predictedLabel === 'your') {
                            sentence += ' are';
                            previousWord = 'are';
                        } else {
                            sentence += ' ' + predictedLabel;
                            previousWord = predictedLabel;
                        }
                        
                        lastPredictionTime = currentTime;
                        liveOutput.value = sentence.trim();
                        
                        // Check for chatbot response
                        checkChatbotResponse(sentence.trim().toLowerCase());
                    }
                }
            }
            
            // Continue processing frames
            requestAnimationFrame(processFrame);
        })
        .catch(error => {
            console.error('Error:', error);
            requestAnimationFrame(processFrame);
        });
    }
    
    function checkChatbotResponse(query) {
        // Simple client-side response check
        const responses = {
            "how are you": "I'm doing great! 😊",
            "what are you doing": "I'm helping you translate signs!",
            "did you eat": "Yes, I had some data bytes! 😄",
            "where are you going": "I'm staying right here to assist you!",
            "what is your name": "I'm APAC, your sign language assistant!"
        };
        
        const cleanedQuery = query.toLowerCase().trim();
        if (responses.hasOwnProperty(cleanedQuery)) {
            chatbotResponse.textContent = `🤖 APAC: ${responses[cleanedQuery]}`;
            chatbotResponse.classList.remove('d-none');
        }
    }
    
    function handleTextToSign() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text first');
            return;
        }
        
        fetch('/text_to_sign', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `text=${encodeURIComponent(text)}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                signImages.innerHTML = '';
                
                if (data.images.length === 0) {
                    signImages.innerHTML = '<div class="col-12"><p>No sign language images found for the entered text.</p></div>';
                    return;
                }
                
                data.images.forEach(image => {
                    const col = document.createElement('div');
                    col.className = 'col-md-4 mb-3';
                    
                    const card = document.createElement('div');
                    card.className = 'card';
                    
                    const img = document.createElement('img');
                    img.src = image.path;
                    img.alt = image.word;
                    img.style.maxWidth = '100%';       // Responsive width
                    img.style.height = 'auto';         // Maintain aspect ratio
                    img.style.objectFit = 'contain';   // Ensure image fits nicely
                    img.style.padding = '10px';        // Optional padding
                    img.className = 'card-img-top';
                    
                    const cardBody = document.createElement('div');
                    cardBody.className = 'card-body';
                    
                    const title = document.createElement('h5');
                    title.className = 'card-title';
                    title.textContent = image.word;
                    
                    cardBody.appendChild(title);
                    card.appendChild(img);
                    card.appendChild(cardBody);
                    col.appendChild(card);
                    signImages.appendChild(col);
                });
            } else {
                signImages.innerHTML = '<div class="col-12"><p>Error processing your request.</p></div>';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            signImages.innerHTML = '<div class="col-12"><p>An error occurred while processing your request.</p></div>';
        });
    }

    
    function copyToClipboard(textarea) {
        textarea.select();
        document.execCommand('copy');
        
        // Show feedback
        const originalText = textarea.value;
        textarea.value = 'Copied to clipboard!';
        setTimeout(() => {
            textarea.value = originalText;
        }, 1000);
    }
    
    // Initialize
    toggleInputSection();

});

// Add this at the bottom of your existing scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle sidebar navigation
    const navLinks = document.querySelectorAll('.sidebar .nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(navLink => {
                navLink.classList.remove('active');
            });
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Get the target section ID from href
            const targetId = this.getAttribute('href').substring(1);
            
            // Hide all input sections
            document.querySelectorAll('.input-section').forEach(section => {
                section.classList.add('d-none');
            });
            
            // Show the selected section
            document.getElementById(targetId).classList.remove('d-none');
            
            // Also update the dropdown to match
            if (targetId === 'uploadSection') {
                document.getElementById('inputChoice').value = 'upload';
            } else if (targetId === 'liveSection') {
                document.getElementById('inputChoice').value = 'live';
            } else if (targetId === 'textSection') {
                document.getElementById('inputChoice').value = 'text';
            }
            
            // For dashboard, you might want to show a welcome screen
            if (targetId === 'dashboard') {
                // Implement your dashboard display logic here
            }
        });
    });
    
    // Make sure the active state matches on page load
    function setInitialActiveState() {
        const currentSection = window.location.hash.substring(1) || 'dashboard';
        document.querySelector(`.nav-link[href="#${currentSection}"]`).classList.add('active');
        
        if (currentSection !== 'dashboard') {
            document.getElementById(currentSection).classList.remove('d-none');
            document.getElementById('inputChoice').value = currentSection.replace('Section', '').toLowerCase();
        }
    }
    
    setInitialActiveState();
});