from tkinter import messagebox
import tkinter as tk


def show_delete(msg):
    root = tk.Tk()
    root.withdraw()
    # messagebox.showerror("Error", msg)
    result = messagebox.askquestion('Delete', msg, icon='warning')

    return result == 'yes'
