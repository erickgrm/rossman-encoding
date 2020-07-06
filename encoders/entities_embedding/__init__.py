''' Entity Embedding Encoder

    author: github.com/erickgrm
'''
try: 
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
except: 
    raise Exception("Scikit-Learn 0.22.2+ not available")
try: 
    from category_encoders import OneHotEncoder
except:
    raise Exception('Category Encoders package not available')
try:
    import tensorflow.keras as keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Dense, Input, Activation, Concatenate, Lambda
except: 
    raise Exception('Tensorflow 2.x not available')

from .utilities import *

class EntityEmbeddingEncoder(): 
    
    def __init__(self, epochs=100):
        super(EntityEmbeddingEncoder,self).__init__() 
        self.model = keras.Model()
        self.ee_layers_model = keras.Model()
        self.epochs = epochs
        self.batch_size = 64
        self.categorical_var_list = []
        self.ohencoders_dict = {}
        
    def fit(self, df, y, cat_cols=[]):
        ''' Retrieves the model architecture and fits the encoding model
        '''
        df = set_categories(df, cat_cols)
        # Scale numerical variables to [0,1]
        df = scale_df(df)

        self.vartypes(df, cat_cols)
        self.model, y = self.architecture(df,y) # a keras model

        # Split
        df_train, df_val, y_train, y_val = train_test_split(df, y, test_size=0.25, random_state=0)
        
        X_train = self.burst_and_ohencode(df_train)
        X_val = self.burst_and_ohencode(df_val)
        
        # Callbacks for early stopping and saving the best model
        model_checkpoint = keras.callbacks.ModelCheckpoint('model_ee.h5', save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(patience=20)

        self.model.fit(X_train,y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0, 
                       validation_data=(X_val,y_val), 
                       callbacks=[model_checkpoint, early_stopping])

        # Load best model, set encoding layers 'EE_layers'
        self.model = keras.models.load_model('model_ee.h5')
        self.ee_layers_model = keras.Model(inputs=self.model.input, 
                                           outputs=self.model.get_layer('C').output)

        
    def transform(self, df, cat_cols=[]):
        df = set_categories(df, self.categorical_var_list)
        df = scale_df(df)
        X = self.burst_and_ohencode(df)
        return pd.DataFrame(self.ee_layers_model.predict(X))
    

    def fit_transform(self, df, y, cat_cols=[]):
        self.fit(df,y, cat_cols)
        return self.transform(df)
    

    def architecture(self, df, y):
        ''' Sets the architecture of the encoding model,
            depending on the dataset structure
        '''
        inputs = []
        ee_layers = []
        for x in df.columns:
            if x in self.categorical_var_list:
                # number of different categories in column
                k = len(np.unique(df[x])) 

                # A heuristic decision, how many encoding neurons to use
                enc_size = min(max(1, int(np.ceil(k/3))), 30) 

                # Add EE Layer
                input = Input(shape=(k,), name='Var_'+str(x))
                output = Dense(units=enc_size, activation='sigmoid', name='EE_Layer_'+str(x))(input) 
                inputs.append(input)
                ee_layers.append(output)
            else:
                # Add identity layer when the column is numerical
                input = Input(shape=(1,),name='Var_'+str(x))
                output = Lambda(lambda x: x, name='Identity_' + str(x))(input)
                inputs.append(input)
                ee_layers.append(output)
                
        # Append las two dense layers
        last_layers = Concatenate(name='C')(ee_layers)
        last_layers = Dense(1000, activation='relu', name='Dense_1000')(last_layers)
        last_layers = Dense(500, activation='relu', name='Dense_500')(last_layers)
        
        # Define final neuron depending o whether the target is categorical or numerical
        #y = pd.Series(np.concatenate(y.values))
        y = y.values
        if len(np.unique(y)) < 10: # if y has only a few values, treat them as classes
            k = len(np.unique(y))
            last_layers = Dense(k, activation='softmax', name='Softmax')(last_layers)
            
            model = keras.Model(inputs=inputs, outputs=last_layers)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            
            y = LabelEncoder().fit_transform(y)
        else:   # Otherwise, assume y is continuous
            last_layers = Dense(1, activation='sigmoid', name='Sigmoid')(last_layers)
            model = keras.Model(inputs=inputs, outputs=last_layers)
            model.compile(optimizer='adam', loss='mean_squared_error',
                              metrics=['accuracy'])
            
        return model, y        


    def plot_model(self):
        """ Plots the model used by the encoder
        """
        try:
            import graphviz
            import pydot
        except:# ImportError as e:
            print('graphviz or pydot module missing')

        keras.utils.plot_model(self.model)
        print('Plot saved as model.png. To display it in a Jupyter notebook, type:')
        print('from IPython.display import Image\nImage(\'model.png\')')


    def vartypes(self, df, cat_cols):
        """ Checks which variables in df are categorical and 
            fits One Hot Encoders for each
        """
        for x in df.columns:
            if is_categorical(df[x]) or x in cat_cols:
                self.categorical_var_list.append(x)
                self.ohencoders_dict[x] = OneHotEncoder().fit(df[x])


    def burst_and_ohencode(self, df):
        """ Splits df into single columns, and applies the corresponding OHE
            to those that are categorical
        """
        X = []
        for x in df.columns:
            if x in self.categorical_var_list:
                X.append(self.ohencoders_dict[x].transform(df[x]).values)
            else: 
                X.append(df[x].values)
        return X
