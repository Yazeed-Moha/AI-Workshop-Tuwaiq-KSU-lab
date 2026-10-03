# Instructor-supplied document

Add your workshop document as **`data/knowledge.txt`**, encoded as UTF-8.
This folder intentionally does not contain invented workshop content.

Commit the approved `.txt` document before sharing the repository with students.
The default limit is 5 MB. Use a short document with several distinct topics for
the lab. A document longer than 1,000 characters makes chunk overlap visible.

Changing `DATA_PATH` in `.env` allows another `.txt` filename. Re-run `chunk`
and `embed` after changing the source. The test fixture is synthetic and is not
loaded by the application.
