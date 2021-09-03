import os
import time
import logging
import argparse
import subprocess

class DockerManager:
    def __init__(self, docker_id, image_name, file_path):
        self.docker_id = docker_id
        self.image_name = image_name
        self.file_path = file_path  # path to a shell
    
    def _get_port(self):
        return str(self.docker_id)
    
    def _get_docker_name(self):
        return f"{self.image_name}_{self.docker_id}"
    
    def _get_local_port_for_target(self):
        return str(self.docker_id + 12340)
    
    def start_docker(self):
        docker_search = f'docker ps -a --filter name=^/{self._get_docker_name()}$'
        p = subprocess.Popen(docker_search, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        out_str = out.decode()
        print(f'out_str = {out_str}')

        str_split = out_str.strip().split()
        if self._get_docker_name() in str_split:
            print('Remove the old container', out_str)
            docker_command = f'docker rm -f {self._get_docker_name()}'
            os.system(docker_command)
        
        n_cpus = os.cpu_count()
        cpu = self.docker_id % n_cpus
        docker_command = \
            f'docker run --cpuset-cpus="{cpu} -itd' \
            f'--name {self._get_docker_name()}' \
            f'{self.image_name}' \
            f'bash {self.file_path}'
        print(docker_command)
        os.system(docker_command)
        os.system('docker ps -a')
        time.sleep(1)

        return True
    
    def pause_docker(self):
        subprocess.run(['docker', 'pause', self._get_docker_name()])
    
    def unpause_docker(self):
        subprocess.run(['docker', 'unpause', self._get_docker_name()])
    
    def stop_docker(self):
        subprocess.run(['docker', 'stop', self._get_docker_name()])
        subprocess.run(['docker', 'rm', '-f', self._get_docker_name()])

    def get_server_address(self):
        ip = subprocess.run(['docker', 'inspect', '--format', ''])
        return ip, self._get_port()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i',
                        type=str)
    parser.add_argument('--file_path', '-f',
                        type=str)
    parser.add_argument('--id',
                        type=int,
                        default=0)
    parser.add_argument('--n_dockers', '-n',
                        type=int,
                        default=1)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    for i in range(args.n_dockers):
        docker_manager = DockerManager(
            args.id+i, args.image, args.file_path
        )
        docker_manager.start_docker()
