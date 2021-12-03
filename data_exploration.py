import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# check all modules are correctly imported and everything is installed in the
# current environment
print('Modules are imported.\n')

# ignore seaborn warnings
warnings.filterwarnings("ignore")


# first we read the dataset into a pandas dataframe
corona_dataset_csv = pd.read_csv("Datasets/covid19_Confirmed_dataset.csv")
# let's look at the start of the data
print("Confirmed cases dataset:")
print(corona_dataset_csv.head(10))
print("\n\n")

# The shape of the dataframe
corona_dataset_csv.shape # gives back: [10 rows x 104 columns]

# We drop the geographic location columns
corona_dataset_csv.drop(["Lat","Long"],axis=1,inplace=True)

# Our data now looks as follows:
print("Confirmed cases dataset (No location data):")
print(corona_dataset_csv.head(10))
print("\n\n")

# Now let's group the data by Countries instead of regions
corona_dataset_aggregated = corona_dataset_csv.groupby("Country/Region").sum()
print("Confirmed cases by country:")
print(corona_dataset_aggregated.head())
print("\n\n")

# Using this we can plot a time series of confirmed cases in each of the countries
# clear the current plot
plt.clf()
corona_dataset_aggregated.loc["China"].plot(color="hotpink")
corona_dataset_aggregated.loc["Italy"].plot(color="green")
corona_dataset_aggregated.loc["Spain"].plot(color="red")
corona_dataset_aggregated.loc["United Kingdom"].plot(color="blue")

ax = plt.gca()
# set the background color of the graph
ax.set_facecolor("#F5F5DC")
plt.title("Confirmed Infections Time Series", fontweight = 'bold')
plt.legend()
plt.savefig('output/confirmed-infections.png', dpi = 300, bbox_inches = 'tight')
# clear the current plot
plt.clf()


corona_dataset_aggregated.loc['China'].plot()
corona_dataset_aggregated.loc["China"][:3].plot()
corona_dataset_aggregated.loc["China"].diff().plot()
corona_dataset_aggregated.loc["China"].diff().max()
corona_dataset_aggregated.loc["Italy"].diff().max()
corona_dataset_aggregated.loc["Spain"].diff().max()
plt.clf()


countries = list(corona_dataset_aggregated.index)
max_infection_rates = []
for c in countries: 
       max_infection_rates.append(corona_dataset_aggregated.loc[c].diff().max())
corona_dataset_aggregated["max_infection_rate"] = max_infection_rates
corona_dataset_aggregated.head()

corona_data = pd.DataFrame(corona_dataset_aggregated["max_infection_rate"])
# See the maximum infection rate for various countries
print("Maximum infection rate by country:")
print(corona_data.head())
print("\n\n")


# Read the world happiness report data into a data frame
happiness_report_csv= pd.read_csv("Datasets/worldwide_happiness_report.csv")
# Here is how the data looks
print("Worldwide happiness report:")
print(happiness_report_csv.head())
print("\n\n")
# Let's drop some of the columns we don't care about:
useless_cols = ["Overall rank","Score","Generosity","Perceptions of corruption"]
happiness_report_csv.drop(useless_cols,axis=1,inplace=True)


happiness_report_csv.set_index("Country or region",inplace=True)
print(happiness_report_csv.head())

# Now let's join both datasets together

# We see that they both don't have the same shape
corona_data.shape
happiness_report_csv.shape

# So we only keep only those rows that match from both using 'inner' 
data = corona_data.join(happiness_report_csv,how="inner")
print("Joined data on corona confirmed cases and happiness report:")
print(data.head())
print("\n\n")
# The correlation  between columns
print("Pairwise correlation between corona and happiness report:")
print(data.corr())


#####################################################################
# Here we plot GDP against the maximum number number of infections
#####################################################################

# reset current matplot
plt.clf()

# The plot data for each axis
x = data["GDP per capita"]
y = data["max_infection_rate"]

# Give a scatter plot of the data using a log scale for y
sns.scatterplot(x,np.log(y))

# plot a linear regression of the data
sns.regplot(x,np.log(y), scatter_kws = {'color': '#301934'},
            line_kws = {'color':'hotpink'})

# plot axes labels and title
plt.xlabel("GDP per capita")
plt.ylabel("Max infection rate (log scale)")
plt.title("GDP vs Maximum Number of Infections", fontweight = 'bold')

# set the background color of the graph
ax = plt.gca()
ax.set_facecolor("#FFEFFF")

# save our plot as a png
plt.savefig('output/GDP-vs-max-infections.png', dpi = 300, bbox_inches = 'tight')


#####################################################################
# Let's plot the level of Social Support in the happiness survey
# against the maximum infection rate under covid
#####################################################################

# reset current matplot
plt.clf()

# Add figure background color
plt.figure(facecolor='#F5F5DC')

# The plot data for each axis
x = data["Social support"]
y = data["max_infection_rate"]

# set the background color of the graph
ax = plt.gca()
ax.set_facecolor("#CFB997")

# plot scatter of x vs y data
sns.scatterplot(x, y, color = 'green')

# plot axes labels and title
plt.xlabel("Social Support")
plt.ylabel("Maximum Infection Rate")
plt.title("Social Support vs Maximum Number of Infections", fontweight = 'bold')

# save our plot as png
plt.savefig('output/Social-support-vs-max-infections.png', dpi = 300, bbox_inches = 'tight')

#####################################################################
# It is hard to see the data because of the scale so let's look again
# using a log scale for y:
#####################################################################

# reset current matplot
plt.clf()
plt.figure(facecolor='#F5F5DC')
x = data["Social support"]
y = data["max_infection_rate"]
plt.title("Social Support vs Maximum Number of Infections", fontweight = 'bold')
ax = plt.gca()
# set the background color of the graph
ax.set_facecolor("#CFB997")
sns.scatterplot(x, np.log(y), color = 'green')
sns.regplot(x,np.log(y), line_kws = {'color':'hotpink', 'linewidth' : 1}, scatter_kws= {'color': 'green'})

plt.xlabel("Social Support")
plt.ylabel("Maximum Infection Rate (log scale)")
plt.savefig('output/Social-support-vs-log-max-infections.png', dpi = 300, bbox_inches = 'tight')


#####################################################################
# Some further plots which we will not save as output
#####################################################################
# reset current matplot
plt.clf()
x = data["Healthy life expectancy"]
y = data["max_infection_rate"]
sns.scatterplot(x, np.log(y))
sns.regplot(x,np.log(y))


# reset current matplot
plt.clf()
x = data["Freedom to make life choices"]
y = data["max_infection_rate"]
sns.scatterplot(x, np.log(y))
sns.regplot(x,np.log(y))


