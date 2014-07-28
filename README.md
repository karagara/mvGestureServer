mvGestureServer
===============

A small project intended to be run alongside mvGestureExtension

Requires boost (http://www.boost.org/)
Requires websocketpp(http://www.zaphoyd.com/websocketpp)
Requires OpenCV

Also, for this to be useful, you're going to need a webcam.

The GestureServer uses websockets to communicate commands to the chrome extension.
It uses openCV to get video from your webcam, and detects the number of fingers you're holding up.
Based on this, it generates a command and sends it to any websocket that is connected.
