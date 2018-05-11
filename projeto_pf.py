#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta classe deve conter todas as suas implementações relevantes para seu filtro de partículas
"""

from pf import Particle, create_particles, draw_random_sample
import numpy as np
import inspercles # necessário para o a função nb_lidar que simula o laser
import math
from scipy.stats import norm


largura = 775 # largura do mapa
altura = 748  # altura do mapa

# Robo
robot = Particle(largura/2, altura/2, math.pi/4, 1.0)

# Nuvem de particulas
particulas = []

num_particulas = 2000

n_lasers = 10


# Os angulos em que o robo simulado vai ter sensores
angles = np.linspace(0.0, 2*math.pi, n_lasers, endpoint=False)

# Lista mais longa
movimentos_longos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0],
              [0,0,math.pi/12.0], [0, 0, math.pi/12.0], [0, 0, math.pi/12],[0,0,-math.pi/4],
              [-5, 0, 0],[-5,0,0], [-5,0,0], [-10,0,0],[-10,0,0], [-10,0,0],[-10,0,0],[-10,0,0],[-15,0,0],
              [0,0,-math.pi/4],[0, 10, 0], [0,10,0], [0, 10, 0], [0,10,0], [0,0,math.pi/8], [0,10,0], [0,10,0], 
              [0,10,0], [0,10,0], [0,10,0],[0,10,0],
              [0,0,-math.radians(90)],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0]]

# Lista curta
movimentos_curtos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0]]

movimentos_relativos = [[0, -math.pi/3],[10, 0],[10, 0], [10, 0], [10, 0],[15, 0],[15, 0],[15, 0],[0, -math.pi/2],[10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2], 
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2], 
                       [10,0], [0, -math.pi/4], [10,0], [10,0], [10,0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]


movimentos = movimentos_relativos


def cria_particulas(minx=0, miny=0, maxx=largura, maxy=altura, n_particulas=num_particulas):
    """
        Cria uma lista de partículas distribuídas de forma uniforme entre minx, miny, maxx e maxy
    """
    var_x = (maxx - minx)/2
    var_y = (maxy - miny)/2
    var_theta = math.pi/3
    return create_particles(robot.pose(), var_x, var_y, var_theta, n_particulas)
    
def move_particulas(particulas, movimento):
    """
        Recebe um movimento na forma [deslocamento, theta]  e o aplica a todas as partículas
        Assumindo um desvio padrão para cada um dos valores
        Esta função não precisa devolver nada, e sim alterar as partículas recebidas.
        
        Sugestão: aplicar move_relative(movimento) a cada partícula
        
        Você não precisa mover o robô. O código fornecido pelos professores fará isso
        
    """
    dp = 1 # Desvio padrão linear
    dp_theta = 0.0349 # Desvio padrão angular

    mov_dp = np.random.normal(0, dp, 1)[0]
    mov_dp_theta = np.random.normal(0, dp_theta, 1)[0]

    mov_l = 0
    mov_t = 0

    # Ciração do movimento com o desvio padrão
    if movimento[0] != 0:
      mov_l = movimento[0] + mov_dp # Movimeno linear
    if movimento[1] != 0:
      mov_t = movimento[1] + mov_dp_theta # Movimento angular

    for p in particulas:
    	p.move_relative([mov_l, mov_t])

    return particulas
    
def leituras_laser_evidencias(robot, particulas):
    """
        Realiza leituras simuladas do laser para o robo e as particulas
        Depois incorpora a evidência calculando
        P(H|D) para todas as particulas
        Lembre-se de que a formula $P(z_t | x_t) = \alpha \prod_{j}^M{e^{\frac{-(z_j - \hat{z_j})}{2\sigma^2}}}$ 
        responde somente P(Hi|D), em que H é a hi
        
        Esta função não precisa retornar nada, mas as partículas precisa ter o seu w recalculado. 
        
        Você vai precisar calcular para o robo
        
    """
    dp = 14 # Desvio padrão do laser

    lasers_robo = inspercles.nb_lidar(robot, angles) #leitura real
    lista_laser_robo = list(lasers_robo.values())

    lista_somatoria = []
    
    # Aplica a leitura do laser para cada particula
    for p in particulas:
      lasers_particula = inspercles.nb_lidar(p, angles)
      lista_laser_particula = list(lasers_particula.values())
      somatoria = 0


      for i in range(len(lista_laser_robo)):
      	somatoria += norm.pdf((lista_laser_particula[i]), lista_laser_robo[i], dp) # Somatória de Monte Carlo
      	#somatoria += np.exp(-(lista_laser_particula[i]-lista_laser_robo[i])/(2*dp*dp))
      lista_somatoria.append(somatoria)

    for j in range(len(lista_somatoria)):
    	particulas[j].w = lista_somatoria[j]/sum(lista_somatoria) # Normalização e atribuição do novo w

    return
    
    # Voce vai precisar calcular a leitura para cada particula usando inspercles.nb_lidar e depois atualizar as probabilidades

def reamostrar(particulas, n_particulas=num_particulas):
    """
        Reamostra as partículas devolvendo novas particulas sorteadas
        de acordo com a probabilidade e deslocadas de acordo com uma variação normal    
        
        O notebook como_sortear tem dicas que podem ser úteis
        
        Depois de reamostradas todas as partículas precisam novamente ser deixadas com probabilidade igual
        
        Use 1/n ou 1, não importa desde que seja a mesma
    """
    lista_pesos = []

    # Cria a lista com os pesos
    for p in particulas:
    	lista_pesos.append(p.w)
    	p.w = 1 # Reseta o w para 1

    # Redesenha as partículas baseado nos pesos
    particulas = draw_random_sample(particulas, lista_pesos, n_particulas)

    return particulas
