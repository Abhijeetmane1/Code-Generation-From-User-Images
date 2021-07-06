try:
    import tkinter as tk
    import tkinter.ttk  as ttk
    from tkinter import filedialog as tkFileDialog
except:
    import Tkinter as tk
    import ttk
    import tkFileDialog
from time import strftime
from PIL import ImageTk, Image
import webbrowser
from run_this import *


MODEL = 0
def set_It(model):
    global MODEL
    MODEL = model
    print(MODEL)


root = tk.Tk()
root.title("Generate Html")
#2400x1380
#800x460
root.geometry('800x460+0+0')

# Creating Menubar
menubar = tk.Menu(root)

# Adding Model selection
model_selection = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Model', menu = model_selection)
model_selection.add_command(label ='Output', command = lambda : set_It(0))
model_selection.add_command(label ='Input', command = lambda : set_It(1))

# Adding Text Info
about = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='about', menu = about)
about.add_command(label ='Html handwritten sketches to html code', command=None)


canvas = tk.Canvas(root, bg='#000000', width=800, height=460)
canvas.pack()

#270x168
upload_image = ImageTk.PhotoImage(Image.open("images/download1.png").resize((270, 168), Image.ANTIALIAS))
panel = tk.Label(canvas, image = upload_image, bd=0)
panel.place(x=400, y=150, anchor=tk.CENTER)

uploaded_image = ''
uploaded_image_panel = tk.Label(canvas, image = uploaded_image, bd=0)

IMAGE = ''
def select_file():
    global IMAGE
    file = tkFileDialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png", "*.png"),("jpeg","*.jpg")))
    if not file:
        return
    IMAGE = file
    browse_button.place_forget()
    #selected_file.place_forget()
    #upload_button.place_forget()
    panel.place_forget()
    img = ImageTk.PhotoImage(Image.open(file).resize((800, 460), Image.ANTIALIAS))
    uploaded_image_panel.config(image=img)
    root.image=img
    uploaded_image_panel.place(x=400, y=230, anchor=tk.CENTER)
    generate_button.place(x=400, y=463, width=794, anchor=tk.N)
    root.geometry('800x535')
    canvas.config(height=535)
    print(file)
    

browse_button = tk.Button(canvas, width=15, height=1, text='Browse', font=("Comic Sans MS", 20, "bold"), cursor='hand2', command = select_file)
browse_button.place(x=400, y=250, anchor=tk.CENTER)

#selected_file = tk.Label(canvas, width=55, height=3, text='Selected File : ', bd=2, font=(20))
#selected_file.place(x=400, y=255, anchor=tk.CENTER)

def generate():
    global IMAGE, MODEL
    print(MODEL)
    load_n_run(image_file=IMAGE, file=True, model=MODEL)
    
    url = 'file:///' + os.path.abspath('output.html')
    webbrowser.open(url, new=2)  # open in new tab
    browse_button.place(x=400, y=250, anchor=tk.CENTER)
    
    panel.place(x=400, y=150, anchor=tk.CENTER)
    uploaded_image_panel.place_forget()
    generate_button.place_forget()
    root.geometry('800x460')
    canvas.config(height=460)

generate_button = tk.Button(canvas, width=15, height=1, bg='#aaaaff', text='Generate html', font=("Comic Sans MS", 20, "bold"), cursor='hand2', command=generate)
#generate_button.place(x=400, y=330, anchor=tk.CENTER)


# display Menu
root.config(menu = menubar)
root.mainloop()

