from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay


def plot_roc_curves(train_roc_auc, split_roc_auc, valid_roc_auc):
    fig, ax = plt.subplots()
    for name, (roc_auc, auc) in [
        ('train', train_roc_auc),
        ('split', split_roc_auc),
        ('valid', valid_roc_auc),
    ]:
        viz = RocCurveDisplay(
            fpr=[x[0] for x in auc],
            tpr=[x[1] for x in auc],
            roc_auc=roc_auc,
            estimator_name=name,
            pos_label=1.0
        )
        viz.plot(ax=ax, name=name)
    return fig
