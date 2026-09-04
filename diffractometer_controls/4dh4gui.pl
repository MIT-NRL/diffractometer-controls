#!/usr/bin/env perl
use strict;
use warnings;
use File::Path qw(make_path);
use Time::HiRes qw(sleep);

# Configuration
my $screen_name = "4dh4gui";  # Name of the screen session
my $launcher_path = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/launcher.py";
my $launcher_dir = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls";  # Directory containing launcher.py
my $conda_env = "bluesky-server";  # Name of the Conda environment
my $conda_activate = "/home/mitr_4dh4/mambaforge/bin/activate";  # Path to Conda's activate script
my $python_cmd = "python3";  # Python command to run the launcher (will be used after activating Conda)
my $state_dir = "$ENV{HOME}/.local/state/diffractometer-controls";
my $log_path = "$state_dir/4dh4gui.log";

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
    my $screen_check = `/usr/bin/screen -list 2>&1`;
    my $session_match = qr/\d+\.\Q$screen_name\E\s+\((?:Detached|Attached)\)/;
    if ($screen_check =~ /\Q$screen_name\E/) {
        # Check if the session is dead (no process)
        if ($screen_check =~ /\Q$screen_name\E\s+\(Dead\)/i) {
            print "Found dead screen session ($screen_name), cleaning up...\n";
            system("/usr/bin/screen -wipe") == 0 or warn "Failed to clean up dead screen session.\n";
        } elsif ($screen_check =~ $session_match) {
            print "The GUI is already running in a screen session ($screen_name).\n";
            exit 1;
        }
    }

    # Start the GUI in a new screen session
    make_path($state_dir) unless -d $state_dir;
    my $command = "/usr/bin/screen -L -Logfile $log_path -dmS $screen_name /usr/bin/bash -lc 'cd $launcher_dir && source $conda_activate $conda_env && exec $python_cmd $launcher_path'";
    print "Starting the GUI in a screen session ($screen_name)...\n";
    system($command) == 0 or die "Failed to start the GUI: $!\n";

    # A successful screen command only means the session was created.  Give
    # Python/Qt enough time to initialize, then verify that it is still alive.
    sleep 3;
    $screen_check = `/usr/bin/screen -list 2>&1`;
    if ($screen_check !~ $session_match) {
        print_startup_log_tail();
        die "The GUI exited during startup. See $log_path for details.\n";
    }

    print "GUI started successfully. Startup log: $log_path\n";
}

# Function to stop the GUI
sub stop_gui {
    # Check if the screen session is running
    my $screen_check = `/usr/bin/screen -list 2>&1`;
    if ($screen_check !~ /\d+\.\Q$screen_name\E\s+\((?:Detached|Attached)\)/) {
        print "The GUI is not running.\n";
        return;
    }

    # Stop the screen session
    print "Stopping the GUI...\n";
    my $command = "/usr/bin/screen -S $screen_name -X quit";
    system($command) == 0 or die "Failed to stop the GUI: $!\n";

    # Avoid racing a following start during restart.
    for (1 .. 30) {
        $screen_check = `/usr/bin/screen -list 2>&1`;
        last if $screen_check !~ /\d+\.\Q$screen_name\E\s+\((?:Detached|Attached)\)/;
        sleep 0.1;
    }

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
    my $screen_check = `/usr/bin/screen -list 2>&1`;
    if ($screen_check !~ /\d+\.\Q$screen_name\E\s+\((?:Detached|Attached)\)/) {
        print "The GUI is not running.\n";
        exit 1;
    }

    # Attach to the screen session
    print "Attaching to the GUI screen session ($screen_name)...\n";
    my $command = "screen -r $screen_name";
    system($command) == 0 or die "Failed to attach to the GUI console: $!\n";
}

# Print a small diagnostic immediately when a detached GUI dies at startup.
sub print_startup_log_tail {
    return unless -f $log_path;
    open my $log_file, '<', $log_path or return;
    my @lines = <$log_file>;
    close $log_file;
    my $first = @lines > 40 ? @lines - 40 : 0;
    print STDERR "Last lines from $log_path:\n";
    print STDERR @lines[$first .. $#lines] if @lines;
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
