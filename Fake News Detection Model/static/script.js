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
