# Launch it in system services
`cd /lib/systemd/system`
`sudo touch test.service`

# Step I: Create service file such as
```
[Unit]
Description=Ensure file exists on boot
After=multi-user.target

[Service]
ExecStart=/bin/bash bridgelauncher.sh
User=lsts

[Install]
WantedBy=multi-user.target
```

# Step II: launch it in system background process.
`sudo systemctl daemon-reload`
`sudo systemctl enable service_test.service`

### To check if the service is enabled.
`sudo systemctl list-unit-files | grep service_test`

### To disable this service.
`sudo systemctl disable service_test.service`

### To start service
`sudo systemctl start service_test.service`

### To stop service
`sudo systemctl stop service_test.service`

### To check service status
`sudo systemctl status service_test.service`

### To check journal of the logs
`sudo journalctl -xe`
`sudo journalctl -u mascot_op1.service`
