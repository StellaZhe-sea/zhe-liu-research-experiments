"""User-requested navy scientific SVG theme with solid lines."""
import matplotlib.pyplot as plt


def apply_theme(name):
    plt.rcParams.update({'lines.linestyle':'-','grid.linestyle':'-'})
    if name=='navy':
        plt.rcParams.update({'svg.fonttype':'none','path.simplify':False,
            'figure.facecolor':'#020611','savefig.facecolor':'#020611','axes.facecolor':'#040b18',
            'axes.edgecolor':'#6485a8','axes.labelcolor':'#e6edf7','text.color':'#e6edf7',
            'xtick.color':'#bed0e5','ytick.color':'#bed0e5','grid.color':'#6485a8',
            'legend.facecolor':'#040b18','legend.edgecolor':'#6485a8','font.size':10})
        return {'traditional':'#56b4ff','multigrade':'#ff9f43','foreground':'#e6edf7'}
    return {'traditional':'#335cbb','multigrade':'#d95d39','foreground':'#293241'}
