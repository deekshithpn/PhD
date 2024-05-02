import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process.kernels import WhiteKernel

# Load the training data 

data = np.loadtxt('Training.txt', delimiter='\t')
num_input_parameters = 1
X_data = data[:,4:9]
y_data = data[:,9]

print(X_data,y_data)

# Pre-processing training data

#Scaling
scaler = MinMaxScaler()
#scaler = StandardScaler()
X_data_scaled = scaler.fit_transform(X_data)
#X_data_scaled = X

print(X_data_scaled ,y_data)

#Shuffle 

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y_data.size), y_data.size, replace=False)
X_shuffled, y_shuffled = X_data_scaled[training_indices], y_data[training_indices]

print(X_shuffled ,y_shuffled)

X_train = X_shuffled
y_train = y_shuffled

print("X_train:", X_train)
print("y_train:", y_train)

# Load the validation data  

dataV = np.loadtxt('Validation.txt', delimiter='\t')
X_dataV = dataV[:,4:9]
y_dataV = dataV[:,9]

X_data_scaledV = scaler.fit_transform(X_dataV)

X_test = X_data_scaledV
y_test = y_dataV


print("X_test:", X_test)
print("y_test:", y_test)


# Create Gaussian Process model

Ls=[1,1,1,1,1]
kernel = C(1, (1e-2, 1e1)) * RBF(Ls, (0.1, 2))
kernel = kernel + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100,random_state=15)
gp.max_iter_predict = 1000000000000000
gp.fit(X_train, y_train)

# Evaluate the GPR performance in training points

y_train_pred,sigma_train = gp.predict(X_train, return_std=True)

# Calculate mean squared error, mean absolute error, and R-squared
absolute_error = np.abs(y_train - y_train_pred)
squared_error = np.square(y_train - y_train_pred)
    
mse_training = mean_squared_error(y_train, y_train_pred)  # Mean Squared Error
mae_training = mean_absolute_error(y_train, y_train_pred)  # Mean Absolute Error
r2_training = r2_score(y_train, y_train_pred)  # R-squared (Coefficient of Determination)


with open("training_metrics_with_errors.txt", "w") as file:
    # Write headers
    file.write("y_train, y_train_pred, absolute_error, squared_error\n")
    
    # Write data in columns
    for i in range(len(y_train)):
        file.write(f"{y_train[i]}, {y_train_pred[i]}, {absolute_error[i]}, {squared_error[i]}\n")



    # Write the metrics to the file
    file.write(f"Mean Squared Error (MSE): {mse_training}\n")
    file.write(f"Mean Absolute Error (MAE): {mae_training}\n")
    file.write(f"R-squared (R2): {r2_training}\n")

print("Data saved to training_metrics_with_errors.txt")


# Access the kernel_ attribute
fitted_kernel = gp.kernel_

# Now you can use fitted_kernel for further analysis or plotting
print(fitted_kernel)

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gp.kernel_} \n"
    f"Log-likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}"
)


# GPR predictions in new data points
matrix = np.random.rand(1000, 5)
matrix = np.sort(matrix, axis=0) 

# Fix a certain value for all other columns

matrix[:, 0] = 0.6
matrix[:, 2] = 0.6
matrix[:, 3] = 0.6
matrix[:, 4] = 0.6

print(matrix)

X_pred = matrix
y_pred, sigma = gp.predict(X_pred, return_std=True)


# GPR validation in test sets

y_test_pred,sigma_test = gp.predict(X_test, return_std=True)

print("Predicted Rg:",y_test_pred)
print("Actual Rg:",y_test)

# Evaluate GPR performance in validation points

mse_validation = mean_squared_error(y_test, y_test_pred)
mae_validation = mean_absolute_error(y_test, y_test_pred)
r2_validation = r2_score(y_test, y_test_pred)

absolute_error = np.abs(y_test - y_test_pred)
squared_error = np.square(y_test - y_test_pred)


with open("validation_metrics_with_errors.txt", "w") as file:
    # Write headers
    file.write("y_test, y_test_pred, absolute_error, squared_error\n")
    
    # Write data in columns
    for i in range(len(y_test)):
        file.write(f"{y_test[i]}, {y_test_pred[i]}, {absolute_error[i]}, {squared_error[i]}\n")


    # Write the metrics to the file
    file.write(f"Mean Squared Error (MSE): {mse_validation}\n")
    file.write(f"Mean Absolute Error (MAE): {mae_validation}\n")
    file.write(f"R-squared (R2): {r2_validation}\n")

print("Data saved to training_metrics_with_errors.txt")


# Plot results

print(y_train)
print(X_train[:,0])
plt.figure(3)
plt.scatter(X_train[:,0], y_train, color = 'red', label='$Observations$')
#plt.errorbar(X_train[:,0].ravel(), y_train, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(X_pred[:,0], y_pred, color='black', label='Prediction')
plt.fill_between(X_pred[:,0],
        		y_pred - 1.9600 * sigma,
                y_pred + 1.9600 * sigma,
        alpha=.9, fc='g', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-8, 18)
plt.xlabel('$a_{AW}$')
plt.ylabel('Rg')
#plt.xlim(-0.1, 1.2)
plt.gca().set_title(gp.kernel_)
plt.legend(loc='upper left')
plt.savefig('aAW_Rg.png')

plt.figure(figsize=(9,6))
plt.scatter(X_train[:,1], y_train, color = 'red', label='$Observations$')
#plt.errorbar(X_train[:,0].ravel(), y_train, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(X_pred[:,1], y_pred, color='black', label='Prediction')
plt.fill_between(X_pred[:,1],
        		y_pred - 1.9600 * sigma,
                y_pred + 1.9600 * sigma,
        alpha=.9, fc='g', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-8, 18)
plt.xlabel('$a_{BW}$')
plt.ylabel('Rg')
#plt.xlim(-0.1, 1.2)
plt.gca().set_title(gp.kernel_)
plt.legend(loc='upper left')
plt.savefig('aBW_Rg.png')

plt.figure(5)
plt.scatter(X_train[:,2], y_train, color = 'red', label='$Observations$')
#plt.errorbar(X_train[:,0].ravel(), y_train, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(X_pred[:,2], y_pred, color='black', label='Prediction')
plt.fill_between(X_pred[:,2],
        		y_pred - 1.9600 * sigma,
                y_pred + 1.9600 * sigma,
        alpha=.9, fc='g', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-8, 18)
plt.xlabel('$a_{AB}$')
plt.ylabel('Rg')
#plt.xlim(-0.1, 1.2)
plt.gca().set_title(gp.kernel_)
plt.legend(loc='upper left')
plt.savefig('aAB_Rg.png')

plt.figure(6)
plt.scatter(X_train[:,3], y_train, color = 'red', label='$Observations$')
#plt.errorbar(X_train[:,0].ravel(), y_train, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(X_pred[:,3], y_pred, color='black', label='Prediction')
plt.fill_between(X_pred[:,3],
        		y_pred - 1.9600 * sigma,
                y_pred + 1.9600 * sigma,
        alpha=.9, fc='g', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-8, 18)
plt.xlabel('$R_0$')
plt.ylabel('Rg')
#plt.xlim(-0.1, 1.2)
plt.gca().set_title(gp.kernel_)
plt.legend(loc='upper left')
plt.savefig('R0_Rg.png')

plt.figure(7)
plt.scatter(X_train[:,4], y_train, color = 'red', label='$Observations$')
#plt.errorbar(X_train[:,0].ravel(), y_train, dy, fmt='r.', markersize=10, label=u'Observations')
plt.plot(X_pred[:,4], y_pred, color='black', label='Prediction')
plt.fill_between(X_pred[:,4],
        		y_pred - 1.9600 * sigma,
                y_pred + 1.9600 * sigma,
        alpha=.9, fc='g', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-8, 18)
plt.xlabel('$K$')
plt.ylabel('Rg')
#plt.xlim(-0.1, 1.2)
plt.gca().set_title(gp.kernel_)
plt.legend(loc='upper left')
plt.savefig('K_Rg.png')

plt.figure(8)

# Plot y_test_pred against y_test
plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted')
plt.plot(y_test, y_test, color='red', label='Ideal line')  # Plotting ideal line (y_test = y_test_pred)
plt.title('Predicted vs Actual')
plt.xlabel('Rg actual values')
plt.ylabel('Rg predicted values')

# Annotate plot with errors
#plt.text(0.1, 0.7, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, color='green', fontsize=22)
#plt.text(0.1, 0.6, f'MAE: {mae:.2f}', transform=plt.gca().transAxes, color='green', fontsize=22)
plt.text(0.1, 0.6, f'R-squared: {r2_validation:.2f}', transform=plt.gca().transAxes, color='green', fontsize=22)

plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Validation_performance.png')


with open("Prediction.txt", "w") as file:
    file.write("X_train[:,1] y_train X_pred[:,1] y_pred\n")
    for x_train, y_train, x_pred, y_pred in zip(X_train[:, 1], y_train, X_pred[:, 1], y_pred):
        file.write(f"{x_train} {y_train} {x_pred} {y_pred}\n")

# Save the figure

#plt.tight_layout()
#plt.show()

