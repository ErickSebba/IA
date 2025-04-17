import tkinter as tk
import serial

arduino = serial.Serial('COM3', 9600)

def enviar_angulo(valor, servo):
    comando = f"{servo}:{valor}\n"
    arduino.write(comando.encode())
    print(f"Enviado: {comando.strip()}")

root = tk.Tk()
root.title("Controle de Servo")

# Cria sliders para cada servo
for i in range(1, 4):
    slider = tk.Scale(root, from_=0, to=180, orient=tk.HORIZONTAL,
                      label=f"Servo {i}", command=lambda val, s=i: enviar_angulo(val, s))
    slider.pack()

root.mainloop()
