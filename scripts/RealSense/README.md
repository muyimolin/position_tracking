# Realsense Scripts

best_fast_grabber is a grabber which has been taken out and slightly refactored from the server.py in the
`RealSenseNetwork` project. In the future this program will merge functions of a grabber and a client
(meaning it can either grab from hardware or from the network) (if it has not yet been done, open an issue
or mail [Alessio](alessio.rocchi@iit.it) for requests ;)


In order for it to work, it needs the compiled `RealSenseLib` by [Mark Draelos](mark.draelos@duke.edu)
which can be downloaded from [Mark's webpage](http://people.duke.edu/~mtd13/RealSense)
Notice that for Windows, you need to install (or copy) the built library inside the `RealSenseLib` folder
