import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.DataFrame({
    'GFLOPs':[42.5,0.752,0.063,7.74,3.84,0.63],
    'mIOU_EDS': [96.97,96.37,93.36,96.83,96.71,96.52],
    'mIOU_Swir':[91.36,82.86,78.02,88.34,87.58,82.86],
    'Model':["AD-GBC","MambaLiteUNet","ULVM-UNet","U-KAN","PMRNet","U-NeXt"]
})
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.scatterplot(data=df, x='GFLOPs', y='mIOU_EDS',style='Model', hue='Model',s=100)
plt.title('Model FLOPs vs. Accuracy(OpenEDS)')
plt.xlabel('FLOPs (GFLOPs) - Lower is faster')
plt.ylabel('Accuracy (%) - Higher is better')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1,2,2)
sns.scatterplot(data=df, x='GFLOPs', y='mIOU_Swir',style='Model', hue='Model',s=100)
plt.title('Model FLOPs vs. Accuracy(Swir, Ellipse fit)')
plt.xlabel('FLOPs (GFLOPs) - Lower is faster')
plt.ylabel('Accuracy (%) - Higher is better')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
