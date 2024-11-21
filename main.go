// Copyright 2024 The Urobot Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"

	htgotts "github.com/hegedustibor/htgo-tts"
	handlers "github.com/hegedustibor/htgo-tts/handlers"
	voices "github.com/hegedustibor/htgo-tts/voices"
)

// Frame is a video frame
type Frame struct {
	Frame *image.YCbCr
	Thumb image.Image
	Gray  *image.Gray
}

func main() {
	speech := htgotts.Speech{Folder: "audio", Language: voices.English, Handler: &handlers.MPlayer{}}
	speech.Speak("Starting...")

	camera := NewV4LCamera()
	say := make(chan string, 8)
	go func() {
		for s := range say {
			speech.Speak(s)
		}
	}()

	go camera.Start("/dev/video0")
	for img := range camera.Images {
		_ = img
	}
}
