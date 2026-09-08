from src.metrics import classification_metrics, ranking_metrics


def test_classification_metrics():
    result = classification_metrics([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9], 0.5)
    assert result["roc_auc"] == 1.0
    assert result["recall"] == 1.0


def test_ranking_metrics():
    result = ranking_metrics(
        [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.5, 0.6, 0.05],
        ks=(0.2,),
    )
    assert len(result) == 1
    assert result.iloc[0]["clientes"] == 2
