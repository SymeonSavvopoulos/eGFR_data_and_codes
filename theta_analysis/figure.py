import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

# Load the dataset
file_path = 'D:/Dropbox/eGFR_article_data_algorithms-main/theta_analysis02102024/spyder_analysis/theta_data_0210.xlsx'  # Adjust the file path as needed
data = pd.read_excel(file_path)
# Calculate median, standard deviation, and interquartile range for each column
median_a, median_b, median_c = data['a'].median(), data['b'].median(), data['c'].median()
std_a, std_b, std_c = data['a'].std(), data['b'].std(), data['c'].std()
iqr_a = data['a'].quantile(0.75) - data['a'].quantile(0.25)
iqr_b = data['b'].quantile(0.75) - data['b'].quantile(0.25)
iqr_c = data['c'].quantile(0.75) - data['c'].quantile(0.25)

# Calculate the 25th and 75th percentiles for each column
q1_a, q3_a = data['a'].quantile(0.25), data['a'].quantile(0.75)
q1_b, q3_b = data['b'].quantile(0.25), data['b'].quantile(0.75)
q1_c, q3_c = data['c'].quantile(0.25), data['c'].quantile(0.75)

# Perform Kruskal-Wallis test across all three columns
kruskal_result = kruskal(data['a'].dropna(), data['b'].dropna(), data['c'].dropna())

# Perform pairwise Mann-Whitney U tests
mannwhitney_a_b = mannwhitneyu(data['a'].dropna(), data['b'].dropna())
mannwhitney_a_c = mannwhitneyu(data['a'].dropna(), data['c'].dropna())
mannwhitney_b_c = mannwhitneyu(data['b'].dropna(), data['c'].dropna())

# Create a new violin plot with the statistical information in a clearer location
plt.figure(figsize=(10, 8))
parts = plt.violinplot([data['a'].dropna(), data['b'].dropna(), data['c'].dropna()], showmedians=True)

# Set different colors for each violin
colors = ['#32CD32', '#4682B4','#FF6347' ]
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])

# Expanding the violins to show the full data range, not just IQR
for partname in ('cbars', 'cmins', 'cmaxes'):
    vp = parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)

# Add lines for 25th percentile (Q1) and 75th percentile (Q3) for each violin
plt.hlines([q1_a, q1_b, q1_c], [0.75, 1.75, 2.75], [1.25, 2.25, 3.25], colors='blue', linestyles='--', label='Q1 (25th percentile)')
plt.hlines([q3_a, q3_b, q3_c], [0.75, 1.75, 2.75], [1.25, 2.25, 3.25], colors='green', linestyles='--', label='Q3 (75th percentile)')

# Set x and y labels with larger font size and bold font weight

plt.xticks([1, 2, 3], ['No rejections (NR)', 'One rejection (OR)', 'Multiple rejections (MR)'],fontweight='bold')
plt.xticks(fontsize=20, fontname='Times New Roman')
plt.yticks(fontsize=20, fontname='Times New Roman')
plt.title('Violin Plot with IQR Lines (Q1 and Q3)', fontsize=22, fontweight='bold', fontname='Times New Roman')

# Set x and y labels with larger font size and bold font weight
#plt.xlabel('Columns', fontsize=22, fontweight='bold', fontname='Times New Roman')


# Prepare the statistical information

median_iqr_texttA = (
    f"Non rejection (NR) patients:"
)

median_iqr_textA = (
    f"Median = {median_a:.1f}, IQR = {iqr_a:.1f}, Q1 = {q1_a:.1f}, Q3 = {q3_a:.1f}"
)

median_iqr_texttB = (
    f"One rejection (OR) patients:"
)
median_iqr_textB = ( 
    f"Median = {median_b:.1f}, IQR = {iqr_b:.1f}, Q1 = {q1_b:.1f}, Q3 = {q3_b:.1f}"
    )
median_iqr_texttC = (
    f"Multiple rejections (MR) patients:"
)
median_iqr_textC = ( 
    f"Median = {median_c:.1f}, IQR = {iqr_c:.1f}, Q1 = {q1_c:.1f}, Q3 = {q3_c:.1f}"
)

median_iqr_KWtext = (
    f"Kruskal-Wallis:"
)
kruskal_text = f"p-value: {kruskal_result.pvalue:.3e}"

mannwhitney_ttext = (
    f"Mann-Whitney U "
    )

mannwhitney_textA = (
    f"NR vs OR: p={mannwhitney_a_b.pvalue:.3e}"
    )

mannwhitney_textB = ( 
    f"NR vs MR: p={mannwhitney_a_c.pvalue:.3e}"
    )

mannwhitney_textC = (
    f"OR vs MR: p={mannwhitney_b_c.pvalue:.3e}"
)

# Set font properties
font_properties = {'fontsize': 18, 'fontname': 'Times New Roman'}

# Add the statistical text in separate rows for better readability
plt.text(1.05, 0.95, median_iqr_texttA, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0),fontweight='bold')
plt.text(1.05, 0.9, median_iqr_textA, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))
plt.text(1.05, 0.85, median_iqr_texttB, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0),fontweight='bold')
plt.text(1.05, 0.80, median_iqr_textB, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))
plt.text(1.05, 0.75, median_iqr_texttC, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0),fontweight='bold')
plt.text(1.05, 0.7, median_iqr_textC, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))
plt.text(1.05, 0.6, median_iqr_KWtext, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0),fontweight='bold')
plt.text(1.05, 0.55, kruskal_text, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))
plt.text(1.05, 0.45, mannwhitney_ttext, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0),fontweight='bold')
plt.text(1.05, 0.40, mannwhitney_textA, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))
plt.text(1.05, 0.35, mannwhitney_textB, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))
plt.text(1.05, 0.30, mannwhitney_textC, transform=plt.gca().transAxes, **font_properties, verticalalignment='top', bbox=dict(facecolor='white', alpha=0))


plt.ylabel('θ Values', fontsize=22, fontweight='bold', fontname='Times New Roman')

plt.grid(True)
plt.legend(loc='upper right', prop={'size': 14,  'family': 'Times New Roman'})  #'weight': 'bold',
plt.show()