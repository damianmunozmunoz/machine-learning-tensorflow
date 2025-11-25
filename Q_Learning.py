#Importamos las librerías
import numpy as np
import random

#Definimos el laberinto donde 1 es la recompensa, -100 es la penalización y 100 es el fin
maze=np.array([[0,0,1],[-100,-100,0],[0,0,100]])

#Guardamos en m y n los ejes del laberinto
m,n=maze.shape

#4 acciones porque son arriba, abajo, izquierda y derecha
actions_space_size=4

#Inicializamos la Q-table con el tamaño del laberinto y sus posibles acciones
Q_table=np.zeros((m,n,actions_space_size))

#Es el estado inicial del agente, la posición (arriba a la izquierda)
state=np.array([0,0])
#Se define un array con los nombres de las posibles acciones
acciones_string=np.array(["izquierda","derecha","arriba","abajo"])

#Definimos una función de dar un paso que recibe la acción a y el estado actual s
def step(a,s):
    #done es si terminó el episodio, que termina si llegamos a 100 o -100
    done=False
    #Si la acción es 0 disminuye columnas
    if a==0:
        s-=np.array([0,1])
    #Si la acción es 1 aumenta columnas
    if a==1:
        s+=np.array([0,1])
    #Si la acción es 2 disminuye filas
    if a==2:
        s-=np.array([1,0])
    #Si la acción es 3 aumenta filas
    if a==3:
        s+=np.array([1,0])
    
    #Evitamos que se salga de las paredes
    if s[0]<0:
        s[0]=0
    if s[1]<0:
        s[1]=0
    if s[0]==m:
        s[0]-=1
    if s[1]==n:
        s[1]-=1
    
    #Evaluamos el estado de la posición en el laberinto
    r=maze[s[0],s[1]]
    #Si muere o gana se termina el episodio
    if r==100 or r==-100:
        done=True
    return s,r,done

#Define las iteraciones (los pasos) que va a hacer
steps=1000
#Define epsilon
eps=1
#Define epsilon_decay
eps_dec=0.99
#Define alpha y gamma
alpha=0.9
gamma=0.9
#Genera un objeto random
random.seed(0)

#Para todos los steps que pongamos
for _ in range(0,steps):
    #Si epsilon es mayor que un número aleatorio entre 0 y 1 (para el primer intento)
    if eps>random.random():
        #Saca una acción aleatoria entre 0 y 3
        accion=random.randint(0,3)
        
        #A partir de la acción y el estado actual cual sería el estado futuro, su recompensa y si habría terminado el juego
        state_t,reward,done=step(accion,state.copy())
        #Actualizamos la Q-table
        Q_table[state[0],state[1],accion] \
        = (1-alpha)*Q_table[state[0],state[1],accion]+alpha*(reward + gamma*np.max(Q_table[state_t[0],state_t[1],:])) #Aplica la ecuación de Bellman
    #Aquí la acción ya no es aleatoria (ya ha aprendido un poco)
    else:
        #Tomamos la acción que nos dé el máximo de entre todas las posibles acciones
        accion=np.argmax(Q_table[state[0],state[1],:])
        #Se transforma a un número de numpy
        accion=np.int8(accion)
        #Dada la acción y el estado actual obtenemos el estado futuro, su recompensa y si habría terminado el juego
        state_t,reward,done=step(accion,state)
        #Actualizamos la Q-table
        Q_table[state[0],state[1],accion] \
        = (1-alpha)*Q_table[state[0],state[1],accion]+alpha*(reward + gamma*np.max(Q_table[state_t[0],state_t[1],:])) #Aplica la ecuación de Bellman
    #Si no se ha terminado el episodio
    if not(done):
        #Pasamos que el estado futuro pase a ser el estado actual
        state=state_t.copy()
    #Si sí terminó se reinicializa el laberinto
    else:
        state[0],state[1]=0,0
        reward=maze[0,0]
        done=False
        #Se suma epsilon_dec para que la probabilidad de que entre al primer else sea mayor
        eps*=eps_dec

#Así sacamos los valores máximos para hayar cual es el camino más óptimo con las posibles opciones en cada caso
value=np.max(Q_table,axis=2)  

#La política es el conjunto de acciones que nos llevan a encontrar la recompensa máxima
policy=np.argmax(Q_table,axis=2)
#Si el campo minado ha terminado se pone la política a -1
policy[maze==100]=-1
policy[maze==-100]=-1
#Muestra las acciones con palabras en la matriz en cada posición
policy_string=acciones_string[policy]
#Si gana
policy_string[maze==100]='Win'
#Si pierde
policy_string[maze==-100]='x'
        