"""Launch the source checkout without installation or working-directory changes."""
def main():
    from diffractometer_controls.launcher import main as launch
    launch()

if __name__ == "__main__":
    main()
