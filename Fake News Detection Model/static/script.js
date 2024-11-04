document.getElementById('predictButton').onclick = function() {
    const newsText = document.getElementById('newsText').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `news_text=${encodeURIComponent(newsText)}`
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `The news is predicted to be: ${data.result}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
};

// copy to clipboard functionality 
function copyToClipboard(text) {
    // Create a temporary textarea element
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    
    // Select the text and copy it
    textarea.select();
    document.execCommand('copy');
    
    // Remove the textarea element
    document.body.removeChild(textarea);
    
    // Optional: Notify the user
    alert('News text copied to clipboard!');
}
