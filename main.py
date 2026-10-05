import csv
import random
import math


random.seed(42)



def load_data(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            if row:
                for i in range(0,4):
                    row[i] = float(row[i])
                separated_data = (row[:4], row[-1])
                data.append(separated_data)

    return data



main_data = load_data('iris.data')




def split_data(data,test_ratio):
    copied_data = data.copy()
    random.shuffle(copied_data)
    split_index = int(len(copied_data)*test_ratio)
    test_data = copied_data[:split_index]
    training_data = copied_data[split_index:]

    return training_data,test_data




train_data,test_data = split_data(main_data,0.2)



def euclidean_distance(a,b):
    a_b = list(zip(a,b))
    
    distance = 0
    for i in a_b:
        distance += (i[0] - i[1])*(i[0] - i[1])

    distance = math.sqrt(distance)
    return distance
            
      


def get_neighbors(train_data,new_point,k):
    data = []
    for i in train_data:
        distance = euclidean_distance(i[0], new_point)
        data.append((distance, i[1]))
    data.sort()
    sorted_data = data[:k]
    return sorted_data   
    
    
    
    

test = get_neighbors(train_data,test_data[0][0],5)





def majority_vote(neighbors):
    top_k_items = {}

    for distance,name in neighbors:
        
        if name in top_k_items:
            top_k_items[name] += 1
        else:
            top_k_items[name] = 1

    return max(top_k_items, key=top_k_items.get)



class KNN:
    def __init__(self,k):
        self.k = k
        self.train_data = []

    def fit(self,train_data):
        self.train_data = train_data

    def predict(self,new_point):
        
        neighbors = get_neighbors(self.train_data, new_point, self.k)
        return majority_vote(neighbors)








model = KNN(5)
model.fit(train_data)



def calculate_accuracy(model,test_data):
    counter = 0
    for numbers,name in test_data:
        predicted_name = model.predict(numbers)
        if predicted_name == name:
            counter += 1

    return counter/len(test_data)



num_seeds = 10
k_values = range(1, 16)
totals = {k: 0 for k in k_values}

for seed in range(1, num_seeds + 1):
    random.seed(seed)
    train, test = split_data(main_data, 0.2)
    for k in k_values:
        model = KNN(k)
        model.fit(train)
        totals[k] += calculate_accuracy(model, test)

for k in k_values:
    print(f"k={k}: {totals[k] / num_seeds * 100:.1f}%")