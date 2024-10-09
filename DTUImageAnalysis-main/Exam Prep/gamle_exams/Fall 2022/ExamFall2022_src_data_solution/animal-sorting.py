import math
import math

cows = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]

# Compute the mean intensity for cows and sheep
mean_cows = sum(cows) / len(cows)
mean_sheep = sum(sheep) / len(sheep)

# Compute the intensity threshold as the midpoint between the two means
threshold = (mean_cows + mean_sheep) / 2

print("Intensity threshold:", threshold)

# Define the parameters for the Gaussian distribution
cows_mean = mean_cows
cows_stddev = math.sqrt(sum((x - cows_mean) ** 2 for x in cows) / len(cows))
sheep_mean = mean_sheep
sheep_stddev = math.sqrt(sum((x - sheep_mean) ** 2 for x in sheep) / len(sheep))

# Compute the values of the Gaussians for value=38
cows_gaussian = (1 / (cows_stddev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((38 - cows_mean) / cows_stddev) ** 2)
sheep_gaussian = (1 / (sheep_stddev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((38 - sheep_mean) / sheep_stddev) ** 2)

print("Cows Gaussian value:", cows_gaussian)
print("Sheep Gaussian value:", sheep_gaussian)