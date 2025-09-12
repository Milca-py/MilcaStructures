import subprocess
import os

def compile_latex(tex_file="ruta"):
    # Compila con pdflatex (dos veces para referencias/índice si hace falta)
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file], check=True)
        
        print("✅ Compilación completada")
        
        # Nombre del PDF esperado
        pdf_file = tex_file.replace(".tex", ".pdf")
        
        # Abrir el PDF (depende del SO)
        if os.name == "nt":  # Windows
            os.startfile(pdf_file)
        elif os.name == "posix":  # Linux/Mac
            subprocess.run(["xdg-open", pdf_file], check=False)  # Linux
            subprocess.run(["open", pdf_file], check=False)      # Mac
        
    except subprocess.CalledProcessError:
        print("❌ Error en la compilación. Revisa los logs.")

# Usar función
compile_latex("milcapy/report/main.tex")
