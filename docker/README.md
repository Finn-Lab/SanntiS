# SanntiS container

#### Get InterProsScan data:
##### The size of download file is ~ 24G, the final directory is 16G. Be sure to have enough space
```bash
$ bash ./get_ips_slim.sh
```

#### Docker ready to use script:
##### Only works if "data/" and sanntis_container.py are in the same directory
```bash
$ sanntis_container.py --help
$ sanntis_container.py [OPTIONS] ARGUMENTS
```

#### Docker image shell:
```bash
$ docker -it --entrypoint bash -v <path to SanntiS/docker>/data/:/opt/interproscan quay.io/repository/microbiome-informatics/sanntis
$ sanntis --help
$ sanntis [OPTIONS] ARGUMENTS
```
