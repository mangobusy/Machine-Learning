import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
DATASET_PATH = "data.csv"
PREV_MODEL_PATH = None

# ----------------- Basic Neural Network Definition -----------------
class Parameter():  # record all parameters in the model
    def __init__(self, data:np.ndarray, requires_grad:bool=True)->None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
    def zero_grad(self)->None:  # set the gradient to zero
        self.grad = None
    def requires_grad_(self, requires_grad:bool=True)->None: # set the requires_grad to the given value
        self.requires_grad = requires_grad

class Module():  # define the basic module
    def forward(self, *input)->np.ndarray:
        raise NotImplementedError
    def backward(self, *gradwrtoutput:np.ndarray)->np.ndarray:
        raise NotImplementedError
    def __call__(self, *args, **kwds): # call the forward function
        return self.forward(*args, **kwds)
    def parameters(self)->list:  # return all the params in the model
        self_params = [v for k, v in self.__dict__.items() if isinstance(v, Parameter)]
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                self_params += v.parameters()
        return self_params
    def trainable_parameters(self)->list:  # get params that needs grad
        return [p for p in self.parameters() if p.requires_grad]
        
class Linear(Module):  # define the linear layer
    def __init__(self, input_size:int, output_size:int, bias:bool=True)->None:
        W = np.random.randn(input_size, output_size)/100  # input_size x output_size
        b = np.zeros(output_size)  # output_size
        self.X = None
        self.W = Parameter(W)
        self.b = Parameter(b)
        if not bias:
            self.b = None
        self.bias = bias
    def forward(self, X:np.ndarray)->np.ndarray:  # forward propagation of linear layer
        self.X = X # bsz x input_size
        l2_norm = np.linalg.norm(self.W.data) # regularisation: calculate the l2 norm of W
        lambda_ = 0
        if self.bias:
            return np.dot(X, self.W.data) + self.b.data + lambda_*l2_norm 
            # return np.dot(X, self.W.data) + self.b.data
        else:
            return np.dot(X, self.W.data) + lambda_*l2_norm
            # return np.dot(X, self.W.data)
    def backward(self, grad:np.ndarray)->np.ndarray:  # backward propagation of linear layer
                                                    # grad is the gradient of upper layer
        """
        loss = f(WX + b), let y= WX + b
        dloss/dW = dloss/dy * dy/dW
        dloss/db = dloss/dy * dy/db
        dloss/dX = dloss/dy * dy/dX
        dloss/dy = grad
        dy/dW = X??
        | w01 w02 w03 |   | x1 |   | y1 | | dloss/dy1 |   
        | w11 w12 w13 | * | x2 | = | y2 | | dloss/dy2 |
        | w21 w22 w23 |   | x3 |   | y3 | | dloss/dy3 |
        d (y) / d (w13) = d y1 / d w13 + d y2 / d w13 + d y3 / d w13 = x3
        -> d (y) / d (W) = X.T.expand(grad.shape[0], -1)
        consider the batch size, we need to sum the gradient of each sample
        dy/db = 1^{input_size}
        """
        # self.X = bsz x input_size
        # grad: bsz x output_size
        self.W.grad = np.dot(self.X.T, grad)  # calculate the gradient of W in this linear layer
        l2_norm_grad = np.linalg.norm(self.W.grad) # regularisation
        lambda_ = 0
        if self.bias:
            self.b.grad = np.sum(grad, axis=0)
        return_grad = np.dot(grad, self.W.data.T)
        return return_grad + lambda_*l2_norm_grad 
        # return return_grad
    def __repr__(self)->str:
        return f"Linear({self.W.data.shape}, {self.b.data.shape})"

   
class RELU(Module):  # define the Relu layer
    def __init__(self)->None:
        self.X = None
    def forward(self, X:np.ndarray)->np.ndarray:
        self.X = X
        return np.maximum(X, 0)
    def backward(self, grad:np.ndarray)->np.ndarray:
        return grad * (self.X > 0)
    
class Softmax(Module):  # define the softmax layer
    def __init__(self)->None:
        self.X = None
    def forward(self, X:np.ndarray)->np.ndarray:
        self.X = X # bsz x d
        exp = np.exp(X) # bsz x d
        self.softmax_X =  exp / np.sum(exp, axis=1, keepdims=True) # bsz x d / bsz x 1 -> bsz x d
        return self.softmax_X
    def backward(self, grad:np.ndarray)->np.ndarray:
        """
        softmax(x) = exp(x) / sum(exp(x))
        d(softmax(x))/d(x) =( d(exp(x))/d(x) * sum(exp(x)) - exp(x) * d(sum(exp(x)))/d(x) ) / (sum(exp(x)))^2
        = exp(x) * (sum(exp(x)) - exp(x)) / (sum(exp(x)))^2
        = softmax(x) * (1 - softmax(x))
        dloss/dx = dloss/d(softmax(x)) * d(softmax(x))/d(x)
        """
        local_grad = self.softmax_X * (1 - self.softmax_X) # bsz x d
        return grad * local_grad
    def __repr__(self)->str:
        return "Softmax()"
    
class CrossEntropyLoss(Module):
    def __init__(self, eps = 1e-8)->None:
        self.y = None
        self.y_hat = None
        self.eps = eps
    def forward(self, y:np.ndarray, y_hat:np.ndarray)->np.ndarray:
        y = np.eye(3)[y] # bsz x 3
        self.y = y # bsz
        self.y_hat = y_hat # bsz x d, after softmax
        CE_loss = -np.sum(y * np.log(y_hat)) / y.shape[0]
        return CE_loss
    def backward(self)->np.ndarray:
        """
        loss = -sum(y * log(y_hat + eps)) / bsz
        dloss/dy_hat = -y / (y_hat + eps) / bsz
        """
        return -self.y / ((self.y_hat) * self.y.shape[0])

# ----------------- Put all layers Together ----------------------------------------------------------
class MLP(Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int)->None:
        self.module_list = [    # define the structure of nn
            Linear(input_size, hidden_size),
            RELU(),
            Linear(hidden_size, output_size),
            Softmax()
        ]
        self.lossfn = CrossEntropyLoss()
    def forward(self, X:np.ndarray, y:np.ndarray=None, return_loss:bool=False)->np.ndarray:
        # Normalization
        X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)    
        for module in self.module_list:  
            X = module(X)  # do forward propagation by calling each layer's forward function
        if return_loss:
            assert y is not None, "y must be provided if return_loss is True"
            y_hat = X
            loss = self.lossfn(y, y_hat)   # calculate the loss
            return y_hat, loss
        return X, None
    def backward(self)->None:
        grad = self.lossfn.backward()
        for module in self.module_list[::-1]:
            grad = module.backward(grad)
    def parameters(self) -> list:
        self_params = []
        for module in self.module_list:
            self_params += module.parameters()
        return self_params
 
# ----------------- Training --------------------------------------------
class SGD():
    def __init__(self, parameters:list, lr:float = .01)->None:
        if len(parameters) == 0:  # check if the parameters are empty
            raise ValueError("parameters should not be empty")
        else:
            # enumerate the number of parameters
            count = 0
            for p in parameters:
                count += 1
                if not isinstance(p, Parameter):
                    raise ValueError(f"Expect Parameter, got {type(p)}")
        self.parameters = parameters
        self.lr = lr
    def step(self)->None:   # do gradient descent if param needs grad
        for p in self.parameters:
            if p.requires_grad:
                p.data -= self.lr * p.grad
    def zero_grad(self)->None:  # clear the grad
        for p in self.parameters:
            if p.requires_grad:
                p.zero_grad()
            assert p.grad is None

class Scheduler():  # adaptive lr: warm-up decay scheduler
    def __init__(self, optimizer:SGD, warmup_steps:int, max_lr:float, decay_steps:int, weight_decay:float = 0.96)->None:
        self.optimizer = optimizer
        self.optimizer.lr = 0
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.current_step = 0
    def step(self)->None:
        if self.current_step < self.warmup_steps:  # if not reach the max_lr
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:     # after reaching the max_lr
            lr = self.max_lr * (self.weight_decay ** ((self.current_step - self.warmup_steps) / self.decay_steps))
        self.optimizer.lr = lr
        self.current_step += 1
        #print(f"Current learning rate: {lr}")
    

class Trainer():
    def __init__(self, model:MLP, optimizer:SGD, scheduler:Scheduler)->None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
    def train(self, data)->float:
        X, y = data
        self.optimizer.zero_grad()
        y_hat, loss = self.model(X, y, return_loss=True)
        acc = np.mean(np.argmax(y_hat, axis=1) == y)
        self.model.backward()
        self.optimizer.step()
        # print(self.optimizer.lr)
        self.scheduler.step()
        #print(self.model.parameters())
        return loss, acc, self.optimizer.lr
    def eval(self, X:np.ndarray, y:np.ndarray)->float:
        y_hat, loss = self.model(X, y, return_loss=True)
        y_hat = np.argmax(y_hat, axis=1)
        acc = np.mean( y_hat == y)
        return loss, acc, y_hat

# ----------------- Data Preprocessing -----------------------------------------------------
class Dataset():   # we read the data from the csv file using pandas
    def __init__(self, csv_path:str)->None:
        data = pd.read_csv(csv_path)
        data = data.dropna()    # drop the NAN value
        self.data = data.sample(frac=1)   # shuffle the data
        self.X = self.data.iloc[:, 1:].values    # preprocess the data by stacking the features and labels
        self.y = self.data.iloc[:, 0].values
    def __len__(self)->int:
        return len(self.data)
    def __getitem__(self, idx:int)->tuple:
        return self.X[idx], self.y[idx]

class TrainTestSplit():   # we split the data into training, testing set, and validation set
    def __init__(self, dataset:Dataset, test_size:float = .2)->None:
        self.dataset = dataset
        self.test_size = test_size
    def collate_fn(self, batch:list[list])->tuple:   # batch: list of list, each list contains a sample and its label
        X, y = zip(*batch)
        return np.array(X), np.array(y) - 1   
    def CrossValidation(self, fold:int)->list:   # cross validation
        n = len(self.dataset)
        test_data = self.collate_fn([self.dataset[idx] for idx in range(161, n)]) # split the testing set
        n = len(self.dataset)-len(test_data[0])
        fold_size = n // fold
        folds = []
        for i in range(fold):
            if i != fold - 1:
                train_data = self.collate_fn([self.dataset[idx] for idx in range(0, i * fold_size)] + [self.dataset[idx] for idx in range((i + 1) * fold_size, n)])
                val_data = self.collate_fn([self.dataset[idx] for idx in range(i * fold_size, (i + 1) * fold_size)])
            else:      # the last fold
                train_data = self.collate_fn([self.dataset[idx] for idx in range(0, i * fold_size)])
                val_data = self.collate_fn([self.dataset[idx] for idx in range(i * fold_size, n)])
            folds.append((train_data, val_data))
        return folds,test_data
    
class DataLoader():
    def __init__(self, dataset:Dataset, batch_size:int)->None:
        self.batch_size = batch_size
        self.dataset = dataset
    def __iter__(self):  # return a data batch each calling
        for i in range(0, len(self.dataset), self.batch_size):
            end_idx = min(i + self.batch_size, len(self.dataset))
            yield [self.dataset[idx] for idx in range(i, end_idx)]
    def __len__(self)->int:
        return len(self.dataset[0]) // self.batch_size

# ----------------- Training -----------------
def train(dataLoader:DataLoader,trainer:Trainer, epochs:int)->None:
    losses= []
    accs = []
    lrs = []
    tbar = tqdm(range(epochs * len(dataLoader)))
    tbar.set_description("Training")
    for epoch in range(epochs):
        total_loss = 0
        for data in dataLoader:
            loss, acc, lr= trainer.train(data)
            total_loss += loss
            losses.append(loss)
            accs.append(acc)
            lrs.append(lr)
            tbar.update(1)
            tbar.set_postfix(loss=loss, acc=acc)
        # print(f"Epoch {epoch}, loss: {total_loss / len(dataLoader)}")
    tbar.close()
    return losses, accs, lrs
def eval(dataLoader:DataLoader, trainer:Trainer)->None:
    total_acc = 0
    total_loss = 0
    for data in dataLoader:
        X, y = data
        loss,acc, y_hat = trainer.eval(X, y)
        total_acc += acc
        total_loss += loss
    # mean_acc = total_acc / len(dataLoader)
    # print(f"Eval loss: {total_loss / len(dataLoader)}, Eval acc: {mean_acc}")
    return total_acc, total_loss, y_hat
def main(): 
    fold_num = 5
    epoch = 1000
    bsz = 20
    hidden_size = 10
    max_lr = 2e-2
    dataset = Dataset(DATASET_PATH)
    train_test_split = TrainTestSplit(dataset)
    fold, test_set = train_test_split.CrossValidation(fold_num)
    # train_data = train_test_split.train_data()
    # test_data = train_test_split.test_data()  
    avg_val_acc = 0
    train_loss = []
    train_acc = []
    lrs = []
    for i in fold:
        train_data, val_data = i[0],i[1]
        dataloader = DataLoader(train_data, bsz)
        model = MLP(13, hidden_size, 3)
        optimizer = SGD(model.trainable_parameters(), lr=.01)
        scheduler = Scheduler(optimizer, 100,max_lr, 500)
        trainer = Trainer(model, optimizer, scheduler)
        if PREV_MODEL_PATH is not None:
            print(f"Loading model from {PREV_MODEL_PATH}")
            trainer.load_model(PREV_MODEL_PATH)
        losses, accs, lr = train(dataloader, trainer, epoch)
        train_loss.append(losses)
        train_acc.append(accs)
        lrs.append(lr)
        val_acc, val_loss, y_hat =  eval(DataLoader(val_data,len(val_data)), trainer)
        avg_val_acc+=val_acc
    avg_val_acc /= fold_num
    print(f"Average validation acc: {avg_val_acc}")

    # test the model
    test_acc, test_loss, y_hat = eval(DataLoader(test_set,len(test_set)), trainer)
    print(f"test acc: {test_acc}")

    # print the confusion matrix
    confusion_mat = confusion_matrix(test_set[1],y_hat)
    print(confusion_mat)


    # plot the loss and accuracy curve in one plot, two y-axis
    fig, ax1 = plt.subplots()
    for i in train_loss:
        ax1.plot(i, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss',color='b')
    ax2 = ax1.twinx()
    for j in train_acc:
        ax2.plot(j, 'r-')
    ax2.set_ylabel('training accuracy', color='r')
    ax2.set_ylim(0, 1)

    # plot the learning rate
    plt.figure()
    for q in lrs:
        plt.plot(q, 'g-')
    plt.ylabel('learning rate', color='g')
    plt.xlabel('epoch')
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
    