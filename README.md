import pygame
import numpy as np
import pyaudio
import random
import math
import time

# ================================
# Inicialización de Pygame y Audio
# ================================
pygame.init()
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Visualizador Avanzado de Música (Audio VLC)")
clock = pygame.time.Clock()

# Configuración de PyAudio para capturar audio de salida (loopback)
CHUNK = 1024
RATE = 44100

p = pyaudio.PyAudio()

# Intentar encontrar un dispositivo WASAPI loopback (o Stereo Mix)
loopback_index = None
wasapi_api_index = None
# Buscar el host WASAPI
for i in range(p.get_host_api_count()):
    host_api_info = p.get_host_api_info_by_index(i)
    if host_api_info['type'] == pyaudio.paWASAPI:
        wasapi_api_index = i
        break

if wasapi_api_index is not None:
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        # El dispositivo debe pertenecer al API WASAPI y tener canales de entrada
        if device_info['hostApi'] == wasapi_api_index and device_info.get('maxInputChannels', 0) > 0:
            # Muchos dispositivos de loopback incluyen la palabra "loopback" en su nombre
            if "loopback" in device_info['name'].lower():
                loopback_index = i
                break

if loopback_index is None:
    print("No se encontró un dispositivo WASAPI loopback. Asegúrate de tener habilitado Stereo Mix o loopback.")
    # Se puede asignar manualmente el índice o dejar None para usar el dispositivo predeterminado
    loopback_index = None

stream = p.open(format=pyaudio.paInt16,
                channels=2,  # Generalmente el audio de salida es estéreo
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=loopback_index)

# ================================
# Variables Globales para Control de Ritmo y Drops
# ================================
last_beat_time = 0.0
current_bpm = 60  # Valor por defecto
beat_min_interval = 0.3  # Mínimo intervalo entre beats (segundos)

last_drop_time = 0.0
drop_cooldown = 2.0  # Tiempo mínimo entre drops (segundos)

smoothed_intensity = None

# Listas para elementos dinámicos
visual_lines = []       # Líneas radiales en primer plano
background_lines = []   # Líneas de fondo que se mueven según BPM
drop_shapes = []        # Figuras especiales que aparecen en drops
particles = []          # Partículas para destellos cinematográficos

# Color base dinámico para algunos efectos
base_color = np.array([random.randint(100, 255) for _ in range(3)], dtype=float)
color_change_speed = 0.05  # Velocidad de cambio del color base

# ================================
# Funciones de Audio y FFT
# ================================
def get_audio_data():
    """
    Lee datos del stream y devuelve un array np.int16.
    Si se capturan 2 canales (estéreo), se promedia para obtener mono.
    """
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Si es estéreo, promedia los dos canales para convertir a mono
        if stream._channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = audio_data.mean(axis=1).astype(np.int16)
        return audio_data
    except Exception as e:
        print("Error leyendo audio:", e)
        return np.zeros(CHUNK, dtype=np.int16)

def get_fft(audio_data):
    """Aplica una ventana Hann y calcula la FFT."""
    window = np.hanning(len(audio_data))
    fft_data = np.abs(np.fft.rfft(audio_data * window))
    return fft_data

# ================================
# Clases para Elementos Visuales
# ================================
# --- Líneas Radiales en Primer Plano ---
class VisualLine:
    def __init__(self, angle, base_length, color):
        self.angle = angle                # Ángulo fijo (grados)
        self.base_length = base_length    # Longitud base mínima
        self.current_length = base_length # Se actualizará con la modulación
        self.color = np.array(color, dtype=float)
    
    def update(self, modulation, smoothing=0.2):
        target_length = self.base_length + modulation
        self.current_length = smoothing * target_length + (1 - smoothing) * self.current_length
    
    def update_color(self, target_color, smoothing=0.1):
        self.color = smoothing * np.array(target_color) + (1 - smoothing) * self.color
    
    def draw(self, surface):
        rad = math.radians(self.angle)
        end_x = int(WIDTH // 2 + self.current_length * math.cos(rad))
        end_y = int(HEIGHT // 2 + self.current_length * math.sin(rad))
        pygame.draw.line(surface, self.color.astype(int), (WIDTH // 2, HEIGHT // 2), (end_x, end_y), 3)

# --- Líneas de Fondo en Movimiento (dependen del BPM) ---
class BackgroundLine:
    def __init__(self):
        self.pos = np.array([random.randint(0, WIDTH), random.randint(0, HEIGHT)], dtype=float)
        self.angle = random.uniform(0, 360)
        self.speed = random.uniform(50, 150)
        self.length = random.randint(100, 300)
        self.color = (random.randint(30, 100), random.randint(30, 100), random.randint(30, 100))
    
    def update(self, dt):
        velocity = self.speed * (current_bpm / 60.0)
        rad = math.radians(self.angle)
        self.pos[0] += velocity * math.cos(rad) * dt
        self.pos[1] += velocity * math.sin(rad) * dt
        if self.pos[0] < -self.length: self.pos[0] = WIDTH + self.length
        if self.pos[0] > WIDTH + self.length: self.pos[0] = -self.length
        if self.pos[1] < -self.length: self.pos[1] = HEIGHT + self.length
        if self.pos[1] > HEIGHT + self.length: self.pos[1] = -self.length
    
    def draw(self, surface):
        rad = math.radians(self.angle)
        end_pos = (int(self.pos[0] + self.length * math.cos(rad)), int(self.pos[1] + self.length * math.sin(rad)))
        pygame.draw.line(surface, self.color, self.pos.astype(int), end_pos, 2)

# --- Partículas para Destellos Cinematográficos ---
class Particle:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(100, 300)
        self.velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.lifetime = random.uniform(0.5, 1.5)
        self.age = 0.0
        self.size = random.randint(4, 8)
        self.color = np.array([255, 255, 255])
    
    def update(self, dt):
        self.age += dt
        self.pos += self.velocity * dt
        self.velocity *= 0.95
    
    def draw(self, surface):
        alpha = max(0, int(255 * (1 - self.age / self.lifetime)))
        if alpha <= 0:
            return
        particle_surface = pygame.Surface((self.size*4, self.size*4), pygame.SRCALPHA)
        for r in range(self.size, 0, -1):
            a = int(alpha * (r / self.size))
            pygame.draw.circle(particle_surface, (255, 255, 255, a), (self.size*2, self.size*2), r)
        surface.blit(particle_surface, (self.pos[0] - self.size*2, self.pos[1] - self.size*2))

# --- Figuras Drop (aparecen en momentos de drop) ---
class DropShape:
    def __init__(self, style, pos):
        self.style = style  # "animal", "math" o "god"
        self.pos = np.array(pos, dtype=float)
        self.creation_time = time.time()
        self.lifetime = 2.0
        self.size = random.randint(100, 200)
    
    def age(self):
        return time.time() - self.creation_time
    
    def is_alive(self):
        return self.age() < self.lifetime
    
    def draw(self, surface):
        t = self.age() / self.lifetime
        alpha = int(255 * (1 - t))
        drop_surface = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        center = (self.size, self.size)
        if self.style == "animal":
            draw_drop_animal(drop_surface, center, self.size, alpha)
        elif self.style == "math":
            draw_drop_math(drop_surface, center, self.size, alpha)
        elif self.style == "god":
            draw_drop_god(drop_surface, center, self.size, alpha)
        rect = drop_surface.get_rect(center=self.pos.astype(int))
        surface.blit(drop_surface, rect)

# ================================
# Funciones para Dibujar Figuras Drop Elaboradas
# ================================
def draw_drop_animal(surface, center, size, alpha):
    pygame.draw.circle(surface, (200, 150, 50, alpha), center, size//2)
    ear_size = size // 4
    left_ear = [(center[0] - size//3, center[1] - size//2),
                (center[0] - size//3 - ear_size//2, center[1] - size//2 - ear_size),
                (center[0] - size//3 + ear_size//2, center[1] - size//2 - ear_size)]
    right_ear = [(center[0] + size//3, center[1] - size//2),
                 (center[0] + size//3 - ear_size//2, center[1] - size//2 - ear_size),
                 (center[0] + size//3 + ear_size//2, center[1] - size//2 - ear_size)]
    pygame.draw.polygon(surface, (150, 100, 30, alpha), left_ear)
    pygame.draw.polygon(surface, (150, 100, 30, alpha), right_ear)
    pygame.draw.circle(surface, (0, 0, 0, alpha), (center[0] - size//8, center[1] - size//8), size//20)
    pygame.draw.circle(surface, (0, 0, 0, alpha), (center[0] + size//8, center[1] - size//8), size//20)

def draw_drop_math(surface, center, size, alpha):
    bar_width = size // 10
    bar_height = size // 2
    left_rect = pygame.Rect(0, 0, bar_width, bar_height)
    left_rect.center = (center[0] - size//4, center[1])
    right_rect = pygame.Rect(0, 0, bar_width, bar_height)
    right_rect.center = (center[0] + size//4, center[1])
    top_rect = pygame.Rect(0, 0, size//2 + bar_width, bar_width)
    top_rect.center = (center[0], center[1] - bar_height//2)
    color = (100, 200, 250, alpha)
    pygame.draw.rect(surface, color, left_rect)
    pygame.draw.rect(surface, color, right_rect)
    pygame.draw.rect(surface, color, top_rect)

def draw_drop_god(surface, center, size, alpha):
    pygame.draw.circle(surface, (250, 220, 200, alpha), center, size//2)
    num_leaves = 8
    for i in range(num_leaves):
        angle = math.pi * (i / (num_leaves - 1))
        leaf_center = (center[0] + int((size//2) * math.cos(angle)),
                       center[1] - int((size//2) * math.sin(angle)))
        pygame.draw.ellipse(surface, (0, 150, 0, alpha), (leaf_center[0]-size//10, leaf_center[1]-size//20, size//5, size//10))

# ================================
# Funciones para Destellos Cinematográficos (Partículas)
# ================================
def update_particles(dt):
    global particles
    new_particles = []
    for particle in particles:
        particle.update(dt)
        if particle.age < particle.lifetime:
            new_particles.append(particle)
    particles = new_particles

def draw_particles(surface):
    for particle in particles:
        particle.draw(surface)

def spawn_particles(pos, count=10):
    for _ in range(count):
        particles.append(Particle(pos))

# ================================
# Función para Dibujar Ondas Circulares (Banda baja)
# ================================
def draw_circular_waves(low_freq, intensity_factor, base_color):
    num_waves = int(5 * intensity_factor)
    for i in range(1, num_waves + 1):
        radius = int(low_freq / 15) + i * 40
        wave_color = (base_color + np.array([i * 10, i * 5, i * 3])) % 255
        pygame.draw.circle(screen, wave_color.astype(int), (WIDTH // 2, HEIGHT // 2), radius, 2)

# ================================
# Función para Destellos Realistas Basados en Altas Frecuencias
# ================================
def draw_high_freq_sparks(high_freq, intensity_factor):
    if high_freq > 5000:
        spawn_particles((WIDTH//2, HEIGHT//2), count=int(5 * intensity_factor))

# ================================
# Dibujo de Visuales en Primer Plano
# ================================
def draw_visuals(fft_data):
    global smoothed_intensity, base_color, visual_lines
    len_fft = len(fft_data)
    low_freq = np.mean(fft_data[:len_fft // 3])
    mid_freq = np.mean(fft_data[len_fft // 3: 2 * len_fft // 3])
    high_freq = np.mean(fft_data[2 * len_fft // 3:])
    overall_intensity = np.mean(fft_data)
    
    if smoothed_intensity is None:
        smoothed_intensity = overall_intensity
    else:
        alpha_smooth = 0.1
        smoothed_intensity = alpha_smooth * overall_intensity + (1 - alpha_smooth) * smoothed_intensity
    
    intensity_factor = min(max(smoothed_intensity / 10000, 0.5), 4.0)
    
    base_color = base_color + color_change_speed * (np.array([high_freq % 50, mid_freq % 50, low_freq % 50]) - 25)
    base_color = np.clip(base_color, 0, 255)
    
    draw_circular_waves(low_freq, intensity_factor, base_color)
    
    modulation = mid_freq / 2
    for line in visual_lines:
        line.update(modulation)
        line.update_color(base_color)
        line.draw(screen)
    
    draw_high_freq_sparks(high_freq, intensity_factor)

# ================================
# Detección de Beat y Drops
# ================================
def detect_beat(overall_intensity, current_time):
    global last_beat_time, current_bpm, smoothed_intensity
    if smoothed_intensity is None:
        smoothed_intensity = overall_intensity
    if overall_intensity > smoothed_intensity * 1.5 and (current_time - last_beat_time) > beat_min_interval:
        interval = current_time - last_beat_time
        current_bpm = 60 / interval if interval > 0 else 60
        last_beat_time = current_time
        spawn_particles((WIDTH//2, HEIGHT//2), count=15)

def detect_drop(overall_intensity, current_time, low_freq, mid_freq, high_freq):
    global last_drop_time
    if overall_intensity > smoothed_intensity * 2.0 and (current_time - last_drop_time) > drop_cooldown:
        last_drop_time = current_time
        if low_freq >= mid_freq and low_freq >= high_freq:
            style = "god"
        elif mid_freq >= low_freq and mid_freq >= high_freq:
            style = "animal"
        else:
            style = "math"
        pos = (WIDTH//2 + random.randint(-200, 200), HEIGHT//2 + random.randint(-200, 200))
        drop_shapes.append(DropShape(style, pos))
        spawn_particles(pos, count=30)

# ================================
# Inicialización de Elementos Visuales
# ================================
for i in range(36):
    angle = i * (360 / 36)
    base_length = 100
    visual_lines.append(VisualLine(angle, base_length, base_color))

for _ in range(5):
    background_lines.append(BackgroundLine())

# ================================
# Fondo Dinámico
# ================================
def draw_dynamic_background(time_elapsed):
    r = int(50 + 50 * math.sin(time_elapsed * 0.1))
    g = int(50 + 50 * math.sin(time_elapsed * 0.1 + 2))
    b = int(50 + 50 * math.sin(time_elapsed * 0.1 + 4))
    bg_color = (r, g, b)
    screen.fill(bg_color)
    for line in background_lines:
        line.draw(screen)

# ================================
# Bucle Principal
# ================================
running = True
while running:
    dt = clock.get_time() / 1000.0
    current_time = time.time()
    time_elapsed = pygame.time.get_ticks() / 1000.0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    audio_data = get_audio_data()
    fft_data = get_fft(audio_data)
    
    len_fft = len(fft_data)
    low_freq = np.mean(fft_data[:len_fft // 3])
    mid_freq = np.mean(fft_data[len_fft // 3: 2 * len_fft // 3])
    high_freq = np.mean(fft_data[2 * len_fft // 3:])
    overall_intensity = np.mean(fft_data)
    
    if smoothed_intensity is None:
        smoothed_intensity = overall_intensity
    else:
        alpha_main = 0.1
        smoothed_intensity = alpha_main * overall_intensity + (1 - alpha_main) * smoothed_intensity
    
    detect_beat(overall_intensity, current_time)
    detect_drop(overall_intensity, current_time, low_freq, mid_freq, high_freq)
    
    for bline in background_lines:
        bline.update(dt)
    
    update_particles(dt)
    drop_shapes = [ds for ds in drop_shapes if ds.is_alive()]
    
    draw_dynamic_background(time_elapsed)
    
    for ds in drop_shapes:
        ds.draw(screen)
    
    draw_visuals(fft_data)
    
    draw_particles(screen)
    
    pygame.display.flip()
    clock.tick(30)

stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
