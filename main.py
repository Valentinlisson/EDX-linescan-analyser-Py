"""Entry point of the analysis suite.

Starts one window holding the three modules, selectable in the tab bar:
  * EDX Line Scan Analyser
  * SEM Picture Analyser
  * LOM Depth Analyser
"""

from edx_analyzer.app import EDXApp

if __name__ == "__main__":
    app = EDXApp()
    app.mainloop()
