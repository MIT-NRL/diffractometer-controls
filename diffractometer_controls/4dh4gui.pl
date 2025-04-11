#!/usr/bin/env perl
use strict;
use warnings;

# Configuration
my $screen_name = "4dh4gui";  # Name of the screen session
my $launcher_path = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/launcher.py";
my $launcher_dir = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls";  # Directory containing launcher.py
my $conda_env = "bluesky-server";  # Name of the Conda environment
my $conda_activate = "/home/mitr_4dh4/mambaforge/bin/activate";  # Path to Conda's activate script
my $python_cmd = "python3";  # Python command to run the launcher (will be used after activating Conda)

# Check for arguments
if (@ARGV < 1) {
    print_usage();
    exit 1;
}

# Parse the command
my $command = shift @ARGV;

if ($command eq "start") {
    start_gui();
} elsif ($command eq "stop") {
    stop_gui();
} elsif ($command eq "restart") {
    stop_gui();
    start_gui();
} elsif ($command eq "run") {
    run_gui();
} elsif ($command eq "console") {
    attach_console();
} else {
    print_usage();
    exit 1;
}

exit 0;

# Function to start the GUI
sub start_gui {
    # Check if the screen session is already running
    my $screen_check = `screen -list | grep $screen_name`;
    if ($screen_check) {
        print "The GUI is already running in a screen session ($screen_name).\n";
        exit 1;
    }

    # Start the GUI in a new screen session
    my $command = "screen -dmS $screen_name bash -c 'cd $launcher_dir && source $conda_activate $conda_env && $python_cmd $launcher_path'";
    print "Starting the GUI in a screen session ($screen_name)...\n";
    system($command) == 0 or die "Failed to start the GUI: $!\n";

    print "GUI started successfully.\n";
}

# Function to stop the GUI
sub stop_gui {
    # Check if the screen session is running
    my $screen_check = `screen -list | grep $screen_name`;
    if (!$screen_check) {
        print "The GUI is not running.\n";
        return;
    }

    # Stop the screen session
    print "Stopping the GUI...\n";
    my $command = "screen -S $screen_name -X quit";
    system($command) == 0 or die "Failed to stop the GUI: $!\n";

    print "GUI stopped successfully.\n";
}

# Function to run the GUI in the foreground
sub run_gui {
    print "Running the GUI in the foreground...\n";
    my $command = "bash -c 'cd $launcher_dir && source $conda_activate $conda_env && $python_cmd $launcher_path'";
    system($command) == 0 or die "Failed to run the GUI: $!\n";

    print "GUI exited.\n";
}

# Function to attach to the screen session
sub attach_console {
    # Check if the screen session is running
    my $screen_check = `screen -list | grep $screen_name`;
    if (!$screen_check) {
        print "The GUI is not running.\n";
        exit 1;
    }

    # Attach to the screen session
    print "Attaching to the GUI screen session ($screen_name)...\n";
    my $command = "screen -r $screen_name";
    system($command) == 0 or die "Failed to attach to the GUI console: $!\n";
}

# Function to print usage instructions
sub print_usage {
    print <<EOF;
Usage: 4dh4gui <command>

Commands:
  start    - Start the GUI in a screen session
  stop     - Stop the GUI screen session
  restart  - Restart the GUI
  run      - Run the GUI in the foreground
  console  - Attach to the GUI screen session
EOF
}