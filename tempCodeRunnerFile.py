@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        keywords = data.get('keywords')
        original_summary = data.get('summary')

        print("Keywords:", keywords)
        print("Original Summary:", original_summary)

        # Searching for related articles
        related_articles = search_related_articles(keywords)
        print("Found related articles:", related_articles)

        related_summaries = [summarize_text(extract_article_content(url)) for _, url in related_articles]
        print("Summarized related articles:", related_summaries)

        similarity_scores = compare_summaries(original_summary, related_summaries)
        print("Similarity scores:", similarity_scores)

        classification = classify_article(similarity_scores)
        print("Classification:", classification)

        contradictions = []
        for summary in related_summaries:
            if check_contradiction(original_summary, summary):
                contradictions.append("Contradiction detected")
            else:
                contradictions.append("No contradiction")

        fact_check_results = get_fact_check_results(keywords)
        print("Fact check results:", fact_check_results)

        # Generate the graph
        graph_path = plot_similarity_graph(similarity_scores, [title for title, _ in related_articles])
        print("Graph path:", graph_path)

        return jsonify({
            "classification": classification,
            "avg_score": round(np.mean(similarity_scores), 2) if similarity_scores else None,
            "articles": [{"title": title, "link": link, "score": round(score, 2), "contradiction": contradiction}
                         for (title, link), score, contradiction in zip(related_articles, similarity_scores, contradictions)],
            "fact_check": fact_check_results,
            "graph_url": "/" + graph_path.replace("\\", "/") if graph_path else None
        })
    
    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({"error": "An error occurred during processing."}), 500

