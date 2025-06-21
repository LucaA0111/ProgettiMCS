import tkinter as tk
from gui import DCTImageCompressor


def main():
    root = tk.Tk()
    app = DCTImageCompressor(root)
    root.mainloop()


if __name__ == "__main__":
    main()