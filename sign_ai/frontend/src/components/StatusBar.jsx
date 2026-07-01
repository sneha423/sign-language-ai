export default function ResultCard({ result, loading, history }) {
  return (
    <section className="panel result-panel">
      <h2>Prediction Result</h2>

      {loading && <p>Analyzing image...</p>}

      {!loading && !result && <p>No prediction yet. Upload or capture an image.</p>}

      {!loading && result?.prediction && (
        <div className="result-box">
          <h3>{result.prediction}</h3>
          <p>
            Confidence:{" "}
            {result.confidence !== null && result.confidence !== undefined
              ? `${(result.confidence * 100).toFixed(2)}%`
              : "N/A"}
          </p>
        </div>
      )}

      {!loading && result?.error && (
        <div className="error-box">
          <p>{result.error}</p>
        </div>
      )}

      <div className="history-box">
        <h3>Recent Predictions</h3>
        {history.length === 0 ? (
          <p>No history yet.</p>
        ) : (
          history.map((item, index) => (
            <div key={index} className="history-item">
              <span>{item.prediction}</span>
              <span>
                {item.confidence !== null && item.confidence !== undefined
                  ? `${(item.confidence * 100).toFixed(1)}%`
                  : "N/A"}
              </span>
              <span>{item.time}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}