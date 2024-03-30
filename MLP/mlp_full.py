import numpy as np
from matplotlib import pyplot
import pandas as pd
from tqdm import tqdm
DATASET_PATH = "data.csv"
PREV_MODEL_PATH = None
BSZ = 142
# this file build a MLP to classify the data into three classes
# we build a 2-layer MLP , with handwritting backpropagation

# ----------------- Basic Neural Network Definition -----------------
class Parameter():
    def __init__(self, data:np.ndarray, requires_grad:bool=True)->None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
    def zero_grad(self)->None:
        self.grad = None
    def __repr__(self)->str:
        return f"Parameter({self.data}, requires_grad={self.requires_grad})"
    def __str__(self)->str:
        return f"Parameter({self.data}, requires_grad={self.requires_grad})"
    def requires_grad_(self, requires_grad:bool=True)->None:
        self.requires_grad = requires_grad
        self.grad = None
class Module():
    def forward(self, *input)->np.ndarray:
        raise NotImplementedError
    def backward(self, *gradwrtoutput:np.ndarray)->np.ndarray:
        raise NotImplementedError
    def zero_grad(self)->None:
        pass
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    def parameters(self)->list:
        self_params = [v for k, v in self.__dict__.items() if isinstance(v, Parameter)]
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                self_params += v.parameters()
        return self_params
    def trainable_parameters(self)->list:
        return [p for p in self.parameters() if p.requires_grad]
    def state_dict(self)->dict:
        state_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter):
                state_dict[k] = v
            if isinstance(v, Module):
                state_dict[k] = v.state_dict()
        return state_dict
    def load_state_dict(self, state_dict:dict, strict:bool=True)->None:
        if strict:
            assert set(self.state_dict().keys()) == set(state_dict.keys()), "Keys of state_dict do not match"
        for k, v in state_dict.items():
            if isinstance(v, Parameter):
                self.__dict__[k] = v
            if isinstance(v, dict):
                self.__dict__[k].load_state_dict(v, strict)
        
class Linear(Module):
    def __init__(self, input_size:int, output_size:int, bias:bool=True)->None:
        W = np.random.randn(input_size, output_size)/100 # input_size x output_size
        #W = np.zeros((input_size, output_size))
        b = np.zeros(output_size) # output_size
        self.X = None
        self.W = Parameter(W)
        self.b = Parameter(b)
        if not bias:
            self.b = None
        self.bias = bias
    def forward(self, X:np.ndarray)->np.ndarray:
        self.X = X # bsz x input_size
        l2_norm = np.linalg.norm(self.W.data)
        lambda_ = 1e-3
        if self.bias:
            return np.dot(X, self.W.data) + self.b.data + lambda_*l2_norm
        else:
            return np.dot(X, self.W.data) + lambda_*l2_norm
    def backward(self, grad:np.ndarray)->np.ndarray:
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
        self.W.grad = np.dot(self.X.T, grad)
        l2_norm_grad = np.linalg.norm(self.W.grad)
        lambda_ = 1e-3
        if self.bias:
            self.b.grad = np.sum(grad, axis=0)
        return_grad = np.dot(grad, self.W.data.T)
        return return_grad + lambda_*l2_norm_grad
    def __repr__(self)->str:
        return f"Linear({self.W.data.shape}, {self.b.data.shape})"
    def zero_grad(self)->None:
        self.W.zero_grad()
        if self.bias:
            self.b.zero_grad()
    def to(self, dtype:str=None)->None:
        if dtype is not None:
            self.W = self.W.to(dtype)
            if self.bias:
                self.b = self.b.to(dtype)
class RELU(Module):
    def __init__(self)->None:
        self.X = None
    def forward(self, X:np.ndarray)->np.ndarray:
        self.X = X
        return np.maximum(X, 0)
    def backward(self, grad:np.ndarray)->np.ndarray:
        return grad * (self.X > 0)
    def zero_grad(self)->None:
        self.X = None
class Softmax(Module):
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
    def zero_grad(self)->None:
        self.X = None
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
    def zero_grad(self)->None:
        self.y = None
        self.y_hat = None

# do regularisation using standard Euclidean norm
# L2 regularisation

# ----------------- Put Together -----------------

class MLP(Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int)->None:
        self.module_list = [
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
            #self.gradient_check(X, module)
            X = module(X)
        if return_loss:
            assert y is not None, "y must be provided if return_loss is True"
            y_hat = X
            loss = self.lossfn(y, y_hat)
            return y_hat, loss
        return X, None
    def gradient_check(self, X:np.ndarray, module:Module)->None:
        # check the gradient of each parameter
        # we use the finite difference method to check the gradient
        # f'(x) = f(x + h) - f(x - h) / 2h
        # we use h = 1e-5
        h_norm = 1e-3
        #test the first dimension of X
        h = np.zeros_like(X)
        h[0, 0] = h_norm
        F_1 = module(X + h).sum()
        F_2 = module(X - h).sum()
        delta_X = (F_1 - F_2) / (2 * h_norm)
        grad = np.ones_like(module(X + h))
        F = module(X)
        print(f"Gradient check for {module}")
        print(f"Analytical gradient: {module.backward(grad)}")
        print(f"Numerical gradient: {delta_X}")
    def backward(self)->None:
        grad = self.lossfn.backward()
        for module in self.module_list[::-1]:
            grad = module.backward(grad)
    def zero_grad(self)->None:
        for module in self.module_list:
            module.zero_grad()
    def to(self, dtype:str=None)->None:
        for module in self.module_list:
            module.to(dtype)
    def parameters(self) -> list:
        self_params = []
        for module in self.module_list:
            self_params += module.parameters()
        return self_params
    def state_dict(self) -> dict:
        state_dict = {}
        for idx, module in enumerate(self.module_list):
            state_dict[f"module_{idx}"] = module.state_dict()
        return state_dict
    def load_state_dict(self, state_dict:dict, strict:bool=True)->None:
        for idx, module in enumerate(self.module_list):
            module.load_state_dict(state_dict[f"module_{idx}"], strict)
# ----------------- Training -----------------
class SGD():
    def __init__(self, parameters:list, lr:float = .01)->None:
        if len(parameters) == 0:
            raise ValueError("parameters should not be empty")
        else:
            # enumerate the number of parameters
            count = 0
            for p in parameters:
                count += 1
                if not isinstance(p, Parameter):
                    raise ValueError(f"Expect Parameter, got {type(p)}")
            # print(f"Number of trainbale parameters: {count}")
        self.parameters = parameters
        self.lr = lr
    def step(self)->None:
        for p in self.parameters:
            if p.requires_grad:
                p.data -= self.lr * p.grad
    def zero_grad(self)->None:
        for p in self.parameters:
            if p.requires_grad:
                p.zero_grad()
            assert p.grad is None

class Scheduler():
    """
    We use a simple warm-up decay scheduler
    """
    def __init__(self, optimizer:SGD, warmup_steps:int, max_lr:float, decay_steps:int, weight_decay:float = 0.95)->None:
        
        self.optimizer = optimizer
        self.optimizer.lr = 0
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.current_step = 0
    def step(self)->None:
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.max_lr * self.weight_decay ** ((self.current_step - self.warmup_steps) / self.decay_steps)
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
        self.scheduler.step()
        #print(self.model.parameters())
        return loss, acc
    def eval(self, X:np.ndarray, y:np.ndarray)->float:
        y_hat, loss = self.model(X, y, return_loss=True)
        acc = np.mean(np.argmax(y_hat, axis=1) == y)
        return loss, acc
    def save_model(self, path:str)->None:
        state_dict = self.model.state_dict()
        print(f"Saving model to {path} with state_dict: {state_dict}")
        np.save(path, state_dict)
    def load_model(self, path:str)->None:
        state_dict = np.load(path, allow_pickle=True).item()
        self.model.load_state_dict(state_dict)

# ----------------- Data Preprocessing ----------------- 
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
        validation_data = self.collate_fn([self.dataset[idx] for idx in range(161, n)]) # split the validation set
        n = len(self.dataset)-len(validation_data[0])
        fold_size = n // fold
        folds = []
        for i in range(fold):
            if i != fold - 1:
                train_data = self.collate_fn([self.dataset[idx] for idx in range(0, i * fold_size)] + [self.dataset[idx] for idx in range((i + 1) * fold_size, n)])
                test_data = self.collate_fn([self.dataset[idx] for idx in range(i * fold_size, (i + 1) * fold_size)])
            else:      # the last fold
                train_data = self.collate_fn([self.dataset[idx] for idx in range(0, i * fold_size)])
                test_data = self.collate_fn([self.dataset[idx] for idx in range(i * fold_size, n)])
            folds.append((train_data, test_data))
        return folds,validation_data
    
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
    tbar = tqdm(range(epochs * len(dataLoader)))
    tbar.set_description("Training")
    for epoch in range(epochs):
        total_loss = 0
        for data in dataLoader:
            loss, acc = trainer.train(data)
            total_loss += loss
            losses.append(loss)
            accs.append(acc)
            tbar.update(1)
            tbar.set_postfix(loss=loss, acc=acc)
        # print(f"Epoch {epoch}, loss: {total_loss / len(dataLoader)}")
    tbar.close()
    return losses, accs
def eval(dataLoader:DataLoader, trainer:Trainer)->None:
    total_acc = 0
    total_loss = 0
    for data in dataLoader:
        X, y = data
        loss,acc = trainer.eval(X, y)
        total_acc += acc
        total_loss += loss
    mean_acc = total_acc / len(dataLoader)
    # print(f"Eval loss: {total_loss / len(dataLoader)}, Eval acc: {mean_acc}")
    return mean_acc
def main(): 
    fold = 5
    dataset = Dataset(DATASET_PATH)
    train_test_split = TrainTestSplit(dataset)
    fold, validation_set = train_test_split.CrossValidation(5)
    # train_data = train_test_split.train_data()
    # test_data = train_test_split.test_data()  
    avg_test_acc = 0
    train_loss = []
    train_acc = []
    for i in fold:
        train_data, test_data = i[0],i[1]
        dataloader = DataLoader(train_data, len(train_data[0]))
        model = MLP(13, 10, 3)
        optimizer = SGD(model.trainable_parameters(), lr=.01)
        scheduler = Scheduler(optimizer, 100, 1e-2, 500)
        trainer = Trainer(model, optimizer, scheduler)
        if PREV_MODEL_PATH is not None:
            print(f"Loading model from {PREV_MODEL_PATH}")
            trainer.load_model(PREV_MODEL_PATH)
        losses, accs = train(dataloader, trainer, 1000)
        train_loss.append(losses)
        train_acc.append(accs)
        avg_test_acc += eval(DataLoader(test_data,len(test_data)), trainer)
    avg_test_acc /= 5
    print(f"Average Test acc: {avg_test_acc}")
    val_acc = eval(DataLoader(validation_set,len(validation_set)), trainer)
    print(f"Validation acc: {val_acc}")
    # plot the loss and accuracy curve in one plot, two y-axis
    fig, ax1 = pyplot.subplots()
    for i in train_loss:
        ax1.plot(i, 'b-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss',color='b')
    ax2 = ax1.twinx()
    for j in train_acc:
        ax2.plot(j, 'r-')
    ax2.set_ylabel('training accuracy', color='r')
    ax2.set_ylim(0, 1)
    # save the plot
    pyplot.savefig("loss_acc.png")
    pyplot.show()
    pyplot.close()
    
    trainer.save_model("model.npy")

if __name__ == "__main__":
    main()
    