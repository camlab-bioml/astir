Development
-----------

How to render the docs
~~~~~~~~~~~~~~~~~~~~~~
Install Sphinx

.. code::

   $ pip install sphinx

From the main project directory `cd` into docs directory

.. code::

   $ cd docs

Build the existing reStructuredText files

.. code::

   $ make html

If the above command causes "Could not import extension <extension-name>"
pip install them until the build succeeds.

Open ``astir/docs/html/index.html`` in your favourite browser either by copying
the absolute path in your browser URL bar.
If you are using `PyCharm` editor, you can

right click on ``index.html``  in the file browser -> `Open in Browser` ->
select your favourite browser



How to run nosetests and add a test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running nosetests
#################

Method 1
********
Run one test module at a time

.. code::

   $ nosetests astir/tests/test_astir.py
   $ nosetests astir/tests/models/test_cellstate.py

Method 2
********
Run all test modules at once

..code::

    $ nosetests

in any project module directory. You might need install the nose package.

Adding a unittest
#################


Best git practices
~~~~~~~~~~~~~~~~~~

The best git practice is to start your own local branch, and commit to your local branch's
remote branch once in awhile. Once your branch is ready to merge into the
origin master repo, you want to git merge, pull, and push.


Git clone and start a new branch
################################

This is the first step you want to take and won't have to repeat unless you want
to clone on another machine or create a new branch.

.. code::

   $ git clone https://github.com/camlab-bioml/astir.git
   $ git checkout -b <new-branch-name>


Update your copy in the repo (git add, commit, push)
####################################################

You might want to do git commits once in awhile to save your work or create new checkpoint.

.. code::

   $ git add <filename1> <filename2> ... <filename n>
   $ git commit -m "<your-commmit-message>"

Additionally push your work in local branch to its remote branch

.. code::

   $ git push origin <my-working-branch-name>

or

.. code::

   $ git push

If you are using the second command make sure that your local branch, called `branch-name`,
is pushing to its remote branch, called `origin/branch-name`


Update origin/master (git merge, pull)
######################################

To update Master remote branch

First, commit and push all your current work to your remote branch

Second, checkout master

.. code::

   $ git checkout master

This changes your working branch to `local master`.

You can view your current working branch with the following command

.. code::

   $ git branch

.. code::

   $ git merge <branch-to-merge-current-with>

Resolve any merge conflicts you get. Once the merge is complete and
all conflicts are resolved

Update the local master branch by

.. code::

   $ git pull origin master

or depending on your setup you may even be able to run

.. code::

   $ git pull

To merge a branch into the current one
Again resolve any conflicts

Update remote master by following the steps outlined in
`Update your copy in the repo`