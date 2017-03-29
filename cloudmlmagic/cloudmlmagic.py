# -*- coding: utf-8 -*-
"""Jupyter Notebook Magic for Google Cloud Machine Learning Engine.
"""
from __future__ import print_function
from IPython.core.magic import (Magics, magics_class, cell_magic, line_magic)
import os
import subprocess
import time
import argparse
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery


@magics_class
class MLMagics(Magics):
    def __init__(self, shell=None,  **kwargs):
        super(MLMagics, self).__init__(shell=shell, **kwargs)
        self._store = []
        shell.user_ns['__mystore'] = self._store

    @line_magic
    def ml_init(self, line):
        """%ml_init
         -projectId PROJECTID
         -bucket BUCKET
         -region us-central1
         -scaleTier BASIC"""

        parser = argparse.ArgumentParser()
        parser.add_argument('-projectId', required=True)
        parser.add_argument('-bucket', required=True)
        parser.add_argument('-region', default='us-central1')
        parser.add_argument('-scaleTier', default='BASIC')

        settings = parser.parse_args(line.split())

        # Store your full project ID in a variable in the format the API needs.
        settings.projectId = 'projects/{}'.format(settings.projectId)

        self.settings = settings

        # Get application default credentials
        # (possible only if the gcloud tool is configured on your machine)
        self.credentials = GoogleCredentials.get_application_default()

        # Build a representation of the Cloud ML API.
        self.ml = discovery.build('ml', 'v1', credentials=self.credentials)

        self.job_id = 'mlmagic__%d' % time.time()

        if not os.path.exists('/tmp/%s/trainer' % self.job_id):
            os.makedirs('/tmp/%s/trainer' % self.job_id)

    @cell_magic
    def ml_code(self, line, cell):
        "store code to deploy"
        self._store.append(cell + '\n')
        self.shell.run_cell(cell)
        return

    @cell_magic
    def ml_run(self, line, cell):
        "%ml_run cloud"

        runoncloud = line == 'cloud'

        self._store.append(cell + '\n')

        if not runoncloud:
            self.shell.run_cell(cell)
            return

        if len(self._store) == 0:
            raise BaseException('Run a code block including model definition')
            return

        tmp_path = '/tmp/%s' % self.job_id

        with open(tmp_path + '/trainer/__init__.py', "w") as f:
            f.write("")
        with open(tmp_path + '/trainer/task.py', "w") as f:
            for r in self._store:
                f.write(r)

        with open(tmp_path + '/setup.py', "w") as f:
            f.write("from setuptools import setup\n"
                    "if __name__ == '__main__':\n"
                    "    setup(name='trainer', packages=['trainer'])\n")

        gzfilepath = tmp_path + '/dist/trainer-0.0.0.tar.gz'
        gsfilepath = 'gs://%s/%s.tar.gz' % (self.settings.bucket, self.job_id)
        subprocess.call(['python', 'setup.py', 'sdist'], cwd=tmp_path)
        subprocess.call(['gsutil', 'cp', gzfilepath, gsfilepath])
        job_req = self.ml.projects().jobs().create(
            parent=self.settings.projectId,
            body={'jobId': self.job_id,
                  'trainingInput': {'scaleTier': 'BASIC',
                                    # 'masterType': 'standard_gpu',
                                    # 'workerType': 'standard_gpu',
                                    'packageUris': [gsfilepath],
                                    'pythonModule': 'trainer.task',
                                    'region': self.settings.region
                                    }
                  }
        )
        response = job_req.execute()
        print(response)
        return


def load_ipython_extension(ipython):
    ipython.register_magics(MLMagics)
