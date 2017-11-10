# -*- coding: utf-8 -*-
"""Jupyter Notebook Magic for Google Cloud Machine Learning Engine.
"""
from __future__ import print_function
from IPython.core.magic import (Magics, magics_class, cell_magic, line_magic, line_cell_magic)
import os
import subprocess
import time
import argparse
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import tempfile
import codecs

@magics_class
class MLMagics(Magics):
    def __init__(self, shell=None,  **kwargs):
        super(MLMagics, self).__init__(shell=shell, **kwargs)
        self._store = []
        shell.user_ns['__mystore'] = self._store

    @line_cell_magic
    def ml_init(self, line, cell=None):
        """%ml_init
         -projectId PROJECTID
         -bucket BUCKET
         -region us-central1
         -scaleTier BASIC
         -runtimeVersion 1.0"""

        parser = argparse.ArgumentParser()
        parser.add_argument('-projectId', required=True)
        parser.add_argument('-bucket', required=True)
        parser.add_argument('-region', default='us-central1')
        parser.add_argument('-scaleTier', default='BASIC')
        parser.add_argument('-runtimeVersion', default='1.0')

        settings = parser.parse_args(line.split())

        # Store your full project ID in a variable in the format the API needs.
        settings.projectId = 'projects/{}'.format(settings.projectId)

        self.settings = settings


        self.ex_settings = {}

        if cell==None:
            ex_settings = None
        else:
            ex_settings = eval(cell)

        if ex_settings != None:
            if type(ex_settings) == dict:
                if 'install_requires' in ex_settings:
                    if not type(ex_settings['install_requires'])==list:
                        raise Exception('Invalid format')
                self.ex_settings = ex_settings
        
        # Get application default credentials
        # (possible only if the gcloud tool is configured on your machine)
        self.credentials = GoogleCredentials.get_application_default()

        # Build a representation of the Cloud ML API.
        self.ml = discovery.build('ml', 'v1', credentials=self.credentials)

        self.job_id = 'mlmagic__%d' % time.time()

        self.tmpdir = tempfile.gettempdir() + '/' + self.job_id
        if not os.path.exists('%s/trainer' % self.tmpdir):
            os.makedirs('%s/trainer' % self.tmpdir)

    @cell_magic
    def ml_code(self, line, cell):
        """Store code to deploy."""
        self._store.append(cell + '\n')
        self.shell.run_cell(cell)
        return

    @cell_magic
    def ml_run(self, line, cell):
        """Run a training job."""
        runoncloud = line == 'cloud'

        self._store.append(cell + '\n')

        if not runoncloud:
            self.shell.run_cell(cell)
            return

        if len(self._store) == 0:
            raise BaseException('Run a code block including model definition')
            return

        with codecs.open(self.tmpdir + '/trainer/__init__.py', "w", 'utf-8') as f:
            f.write("")
        with codecs.open(self.tmpdir + '/trainer/task.py', "w", 'utf-8') as f:
            for r in self._store:
                f.write(r)

        requires = ",".join(["'{}'".format(s) for s in self.ex_settings.get('install_requires','')])
        with codecs.open(self.tmpdir + '/setup.py', "w", 'utf-8') as f:
            f.write("from setuptools import setup\n"
                    "if __name__ == '__main__':\n"
                    "    setup(name='trainer',\n"
                    "          packages=['trainer'],\n"
                    "          install_requires=[{}])\n".format(requires))

        gzfilepath = self.tmpdir + '/dist/trainer-0.0.0.tar.gz'
        gsfilepath = 'gs://%s/%s.tar.gz' % (self.settings.bucket, self.job_id)
        subprocess.call(['python', 'setup.py', 'sdist'], cwd=self.tmpdir)
        subprocess.call(['gsutil', 'cp', gzfilepath, gsfilepath])
        job_req = self.ml.projects().jobs().create(
            parent=self.settings.projectId,
            body={'jobId': self.job_id,
                  'trainingInput': {'scaleTier': self.settings.scaleTier,
                                    # 'masterType': 'standard_gpu',
                                    # 'workerType': 'standard_gpu',
                                    'packageUris': [gsfilepath],
                                    'pythonModule': 'trainer.task',
                                    'region': self.settings.region,
                                    'runtimeVersion': self.settings.runtimeVersion
                                    }
                  }
        )
        response = job_req.execute()
        print(response)
        
        
        from google.cloud import logging as gcloud_log
        from google.cloud.logging import ASCENDING
        from datetime import datetime
        import pytz
        import time
        import sys

        # Instantiates a client
        client = gcloud_log.Client()

        FILTER = 'resource.labels.job_id:' + response['jobId'] + ' AND timestamp > "{0:%Y-%m-%dT%H:%M:%S.%f}Z"'

        dt = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
        is_running = True
        while is_running:
            for entry in client.list_entries(filter_=FILTER.format(dt), order_by=ASCENDING):  # API call(s)
                if dt < entry.timestamp.astimezone(pytz.utc):
                    if type(entry.payload)==dict:
                        msg = entry.payload.get('message')
                    else:
                        msg = entry.payload

                    if entry.severity in ['ERROR', 'CRITICAL']:
                        stout = sys.stderr
                    else:
                        stout = sys.stdout

                    stout.write("({0:%H:%M:%S}): {1}\n".format(entry.timestamp, msg))
                    stout.flush()

                    dt = entry.timestamp
            if msg.startswith(('Job completed', 'Job failed')):
                is_running = False
            else:
                time.sleep(1)
        return


def load_ipython_extension(ipython):
    ipython.register_magics(MLMagics)
