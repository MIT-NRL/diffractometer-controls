/* Foreground IOC with graceful systemd shutdown. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <epicsExit.h>
#include <epicsThread.h>
#include <iocsh.h>

static volatile sig_atomic_t stopping = 0;
static void request_stop(int signal_number) { (void)signal_number; stopping = 1; }

int main(int argc, char *argv[])
{
    int service = argc == 3 && strcmp(argv[1], "--service") == 0;
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    if (service) {
        signal(SIGTERM, request_stop);
        signal(SIGINT, request_stop);
    }
    if (argc >= 2 && iocsh(argv[service ? 2 : 1]) != 0) {
        epicsExit(1);
        return 1;
    }
    if (!service) {
        iocsh(NULL);
    } else {
        while (!stopping) epicsThreadSleep(0.2);
    }
    if (service) {
        /* Run every EPICS cleanup hook (including autosave).  The legacy
         * Newport library's static event destructor waits on its live polling
         * thread during libc exit; avoid that destructor after IOC cleanup. */
        epicsExitCallAtExits();
        fflush(NULL);
        _Exit(0);
    }
    epicsExit(0);
    return 0;
}
