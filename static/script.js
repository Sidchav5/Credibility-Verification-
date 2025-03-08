function processInput() {
    let inputType = document.getElementById("inputType").value;
    let content = document.getElementById("content").value;

    fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputType, content })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("summaryText").innerText = data.summary;
        document.getElementById("summarySection").style.display = "block";
        
        // Display auto-extracted keywords
        document.getElementById("keywords").value = data.keywords.join(", ");
    });
}

function searchArticles() {
    let keywords = document.getElementById("keywords").value.split(",");
    let summary = document.getElementById("summaryText").innerText;

    fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keywords, summary })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("classification").innerText = data.classification;
        document.getElementById("avgScore").innerText = data.avg_score;

        let articleList = document.getElementById("articleList");
        let factCheckList = document.getElementById("factCheckList");
        let similarityGraph = document.getElementById("similarityGraph");
        let suspiciousNote = document.getElementById("suspiciousNote");

        if (data.articles.length > 0) {
            let articles = data.articles.map(a => 
                `<li><a href="${a.link}" target="_blank">${a.title}</a> - Score: ${a.score}</li>`
            ).join("");
            articleList.innerHTML = articles;
            suspiciousNote.style.display = "none";
        } else {
            articleList.innerHTML = "<li>No related articles found.</li>";
            suspiciousNote.style.display = "block";
        }

        if (data.fact_check.length > 0) {
            let factChecks = data.fact_check.map(f => 
                `<li>${f.claim} - <a href="${f.url}" target="_blank">Fact Check</a></li>`
            ).join("");
            factCheckList.innerHTML = factChecks;
        } else {
            factCheckList.innerHTML = "<li>No fact-checks available.</li>";
        }

        if (data.graph_url) {
            similarityGraph.src = data.graph_url;
            similarityGraph.style.display = "block";
        } else {
            similarityGraph.style.display = "none";
        }

        document.getElementById("results").style.display = "block";
    })
    .catch(error => console.error("Error fetching articles:", error));
}

