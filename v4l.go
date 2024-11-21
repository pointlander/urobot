// Copyright 2024 The Ouroboros Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
	"runtime"
	"sort"
	"time"

	"github.com/blackjack/webcam"
	"github.com/nfnt/resize"
)

// FrameSizes is a slice of FrameSize
type FrameSizes []webcam.FrameSize

// Len is the length of the slice
func (slice FrameSizes) Len() int {
	return len(slice)
}

// For sorting purposes
func (slice FrameSizes) Less(i, j int) bool {
	ls := slice[i].MaxWidth * slice[i].MaxHeight
	rs := slice[j].MaxWidth * slice[j].MaxHeight
	return ls < rs
}

// For sorting purposes
func (slice FrameSizes) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

// V4LCamera is a camera that is from a v4l device
type V4LCamera struct {
	Stream bool
	Images chan Frame
}

// NewV4LCamera creates a new v4l camera
func NewV4LCamera() *V4LCamera {
	return &V4LCamera{
		Stream: true,
		Images: make(chan Frame, 1),
	}
}

// Start starts streaming
func (vc *V4LCamera) Start(device string) {
	runtime.LockOSThread()
	skip := 0
	fmt.Println(device)
	camera, err := webcam.Open(device)
	if err != nil {
		panic(err)
	}
	defer camera.Close()

	format_desc := camera.GetSupportedFormats()
	var formats []webcam.PixelFormat
	for f := range format_desc {
		formats = append(formats, f)
	}
	sort.Slice(formats, func(i, j int) bool {
		return format_desc[formats[i]] < format_desc[formats[j]]
	})
	println("Available formats: ")
	for i, value := range formats {
		fmt.Printf("[%d] %s\n", i+1, format_desc[value])
	}
	format := formats[1]

	fmt.Printf("Supported frame sizes for format %s\n", format_desc[format])
	frames := FrameSizes(camera.GetSupportedFrameSizes(format))
	sort.Sort(frames)
	for i, value := range frames {
		fmt.Printf("[%d] %s\n", i+1, value.GetString())
	}
	size := frames[0]

	f, w, h, err := camera.SetImageFormat(format, uint32(size.MaxWidth), uint32(size.MaxHeight))
	if err != nil {
		panic(err)
	} else {
		fmt.Printf("Resulting image format: %s (%dx%d)\n", format_desc[f], w, h)
	}

	err = camera.StartStreaming()
	if err != nil {
		panic(err)
	}
	defer camera.StopStreaming()

	var cp []byte
	start, count := time.Now(), 0.0
	_ = start
	for vc.Stream {
		err := camera.WaitForFrame(5)

		switch err.(type) {
		case nil:
		case *webcam.Timeout:
			fmt.Println(device, err)
			continue
		default:
			panic(err)
		}

		frame, err := camera.ReadFrame()
		if err != nil {
			fmt.Println(device, err)
			continue
		} else {
			//fmt.Println(device)
		}
		count++

		//fmt.Println(device, count/float64(time.Since(start).Seconds()))

		if skip < 20 {
			skip++
		} else {
			skip = 0
			continue
		}
		if len(frame) != 0 {
			if len(cp) < len(frame) {
				cp = make([]byte, len(frame))
			}
			copy(cp, frame)
			//fmt.Printf("Frame: %d bytes\n", len(cp))
			yuyv := image.NewYCbCr(image.Rect(0, 0, int(w), int(h)), image.YCbCrSubsampleRatio422)
			for i := range yuyv.Cb {
				ii := i * 4
				yuyv.Y[i*2] = cp[ii]
				yuyv.Y[i*2+1] = cp[ii+2]
				yuyv.Cb[i] = cp[ii+1]
				yuyv.Cr[i] = cp[ii+3]

			}
			thumb := resize.Resize(uint(w)/16, uint(h)/16, yuyv, resize.NearestNeighbor)
			gray := image.NewGray(thumb.Bounds())
			dx := thumb.Bounds().Dx()
			dy := thumb.Bounds().Dy()
			for x := 0; x < dx; x++ {
				for y := 0; y < dy; y++ {
					gray.Set(x, y, color.GrayModel.Convert(thumb.At(x, y)))
				}
			}

			select {
			case vc.Images <- Frame{
				Frame: yuyv,
				Thumb: thumb,
				Gray:  gray,
			}:
			default:
				//fmt.Println("drop", device)
			}
		}
	}
}
