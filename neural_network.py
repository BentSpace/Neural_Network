import numpy as np


# Helper function to evaluate the total loss on the dataset 
# model is the current version of the model {’W1’:W1,’b1’:b1,’W2’:W2,’b2’:b2’} It’s a dictionary.
# X is all the training data
# y is the training labels
def calculate_loss (model , X, y):
    num_rows, num_cols = X.shape
    num_features = num_cols
    num_samples = num_rows
    L = 0
    i = 0
    while i < num_samples:
        x = X[i]
        y_c = y[i]
        y_hat = predict(model, x)
        l = y_c * np.log(y_hat)
        L += l
        i += 1
    L = L * (-1 / num_samples) 
    return L

# Helper function to predict an output (0 or 1) 
# model is the current version of the model {’W1’:W1,’b1’:b1,’W2’:W2,’b2’:b2’} It’s a dictionary.
# x is one sample (without the label) 
def predict (model, x) : 
    dimensions = x.ndim
    if dimensions == 2:  
        num_rows, num_cols = x.shape
        num_features = num_cols
        num_samples = num_rows
    else:
        num_samples = 1
        
    y_hat = np.zeros((num_samples))
    k = 0
    while k < num_samples:
        a = np.dot(x[k], model.get('W1')) + model.get('b1')
        h = np.tanh(a)
        z = np.dot(h, model.get('W2')) + model.get('b2')
        
        softmax_bottom = np.exp(z[0, 0]) + np.exp(z[0, 1])
        i = 0
        y_hat_single = np.array([0, 0])
        
        while i < 2:
            softmax_top = np.exp(z[0, i])
            y_hat_single[i] = softmax_top / softmax_bottom
            i += 1
        
            
        if y_hat_single[0] > y_hat_single[1]:
            y_hat[k] = 0
        else:
            y_hat[k] = 1 
            
        if np.isnan(y_hat):
            print(f"It's np.isnan  : {np.isnan(y_hat)}")
        k += 1
    return y_hat

# This function learns parameters for the neural network and returns the model.
# − X is the training data 
# − y is the training labels 
# − nn_hdim : Number of nodes in the hidden layer 
# − num_passes : Number of passes through the training data for gradient descent
# − print loss : If True, print the loss every 1000 iterations 
def build_model(X, y, nn_hdim, num_passes=100, print_loss=False): 
    # Intialize our parameter with random values between -0.1 and 0.1
    W1 = np.random.uniform(-0.1, 0.1, (2, nn_hdim))
    W2 = np.random.uniform(-0.1, 0.1, (nn_hdim, 2))
    b1 = np.random.uniform(-0.1, 0.1, (1, nn_hdim))
    b2 = np.random.uniform(-0.1, 0.1, (1, 2))
    model = {'W1': W1, 'b1': b1, 'W2': W2,'b2': b2}
    learning_rate = 0.1
    
    num_rows, num_cols = X.shape
    num_features = num_cols
    num_samples = num_rows
    
    i = 0   
    while i < num_passes:
        if print_loss == True and (i % 1000) == 0:
            total_loss = calculate_loss (model, X, y)
            print(total_loss)
        k = 0
        while k < num_samples:
            
            #x = X[k]
            x_raw = X[k]
            
            # x = np.reshape(x_raw, (1,2))
            
            x = np.array([x_raw])
            
            # if y[k] == 0:
            #     y_label = np.array([[1.0, 0.0]])
            # if y[k] == 1:
            #     y_label = np.array([[0.0, 1.0]])
            
            y_hat = predict(model, x)
            dLdy_hat = y_hat - y[k]
            a = np.dot(x, model.get('W1')) + model.get('b1')
            dLda = (1 - (np.tanh(a)) ** 2) * (dLdy_hat * model.get('W2').transpose())
            h = np.tanh(a)
            dLdW2 = np.dot(h.transpose(), dLdy_hat)
            dLdb2 = dLdy_hat
            x_transposed = x.transpose()
            dLdW1 = np.dot(x_transposed, dLda)
            dLdb1 = dLda
            
            W1_new = model.get('W1') - learning_rate * dLdW1
            model.update({'W1': W1_new})
            
            W2_new = model.get('W2') - learning_rate * dLdW2
            model.update({'W2': W2_new})
            
            b1_new = model.get('b1') - learning_rate * dLdb1
            model.update({'b1': b1_new})
            
            b2_new = model.get('b2') - learning_rate * dLdb2
            model.update({'b2': b2_new})
            
            k += 1
        i += 1
    return model








