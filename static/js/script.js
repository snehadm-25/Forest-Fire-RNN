document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    // Convert string inputs to numbers
    const numericFields = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'];
    numericFields.forEach(field => {
        data[field] = parseFloat(data[field]);
    });

    const submitBtn = document.querySelector('.submit-btn');
    const resultCard = document.getElementById('resultCard');
    const predictionText = document.getElementById('predictionText');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');

    submitBtn.textContent = 'Analyzing...';
    submitBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (result.success) {
            resultCard.classList.remove('hidden');
            predictionText.textContent = result.result;
            
            // Set color based on result
            if (result.result === 'FIRE') {
                predictionText.style.color = '#ff4d4d';
                confidenceBar.style.background = '#ff4d4d';
                resultCard.style.borderLeftColor = '#ff4d4d';
            } else {
                predictionText.style.color = '#22c55e';
                confidenceBar.style.background = '#22c55e';
                resultCard.style.borderLeftColor = '#22c55e';
            }

            confidenceBar.style.width = `${result.confidence}%`;
            confidenceText.textContent = `Confidence: ${result.confidence}%`;

            // Smooth scroll to result
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            alert('Error: ' + result.error);
        }
    } catch (err) {
        alert('Failed to connect to the server.');
    } finally {
        submitBtn.textContent = 'Run AI Analysis';
        submitBtn.disabled = false;
    }
});
