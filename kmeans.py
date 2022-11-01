import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt

def plot(df, feature1, feature2):
    """
    Plots the data
    """
    plt.scatter(df[feature1],df[feature2])   
    plt.xlabel(feature1)
    plt.ylabel(feature2)           
    plt.show()       


def initializeMeans(df, k):
    """
    Intialize the means by splitting the df k times.
    returns k dataframes.
    """
    split = np.array_split(df, k)
    return split

class centroid: # class to store a centroid and the distance to a point
    def __init__(self, number, distance): 
        self.number = number 
        self.distance = distance

def euclideanDist(df,pointIDX,centroids):
    """
    Take the index of the point in the dataframe you want to calculate the 
    distance from and calculate the euclidean distance to all centroids.
    
    returns a pandas dataframe (or pandas series) with the closest centroid assigned to column 'class' given the pointIDX.
    """
    point = df.iloc[[pointIDX]]
    cent = np.array(centroids, dtype=object)
    p = np.array(point)[:,:2]
    
    cent_list = [] #list for centroid and the distances from them to the point
    number = 1
    for x in cent:
        flat = np.array(x).flatten()
        mean = flat[:2]
        distance = np.linalg.norm(p-mean)
        cent_list.append(centroid(number, distance))
        number = number + 1

    minimum = min(cent_list, key=attrgetter('distance'))
    df.at[pointIDX, 'class'] = minimum.number # set class to the number of the clostest centroid

    return df

def updateMean(df, k):
    """
    takes the df containing the assigned classes
    returns updated centroids based on the mean value in column 'class'
    """
    cent_list = []
    x = 1
    for i in range(k):
        newCent = df[df['class'] == x].mean()
        x = x + 1
        cent_list.append(newCent)
    return np.array(cent_list)

def Kmeans(df,iterations, k):
    """
    Works on 2-12 clusters 
    (due to the possible colors in the color array)
    Returns the final dataframe and an array with the centroids
    """
    centroids = initializeMeans(df, k)

    prev = np.array(centroids, dtype=object)

    for iteration in range(iterations):

        print("Iteration {}/{}".format(iteration,iteration))
        
        for i in range(len(df)):
            df = euclideanDist(df,i,centroids)

        if np.array_equal(prev, centroids):
            break
        else:
            prev = centroids
            centroids = updateMean(df, k)

    plt.clf()
    colors = ['b','g','c','m','y','darkorchid','palegreen', 'chocolate', 'lightcoral', 'dodgerblue', 'lime', 'aqua']
    n = 1
    for x in centroids:
        plt.scatter(df.loc[df['class'] == n][df.columns[0]],df.loc[df['class'] == n][df.columns[1]],
        color=colors[n-1],label=n)  
        plt.scatter(x[0],x[1],s=50, marker='s',color='k')                   
        plt.legend() 
        n = n + 1   
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])                                                                                                  
    plt.show()  
            
    return df,centroids

    