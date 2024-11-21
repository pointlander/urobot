// Copyright 2024 The Urobot Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"

	htgotts "github.com/hegedustibor/htgo-tts"
	handlers "github.com/hegedustibor/htgo-tts/handlers"
	voices "github.com/hegedustibor/htgo-tts/voices"
)

const (
	// Batch is the batch size
	Batch = 16
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = .01
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new float64 matrix
func NewMatrix(cols, rows int, data ...float64) Matrix {
	if data == nil {
		data = make([]float64, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// Dot computes the dot product
func dot(x, y []float64) (z float64) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// MakeRandomTransform makes a random transform
func MakeRandomTransform(rng *rand.Rand, cols, rows int, stddev float64) Matrix {
	transform := NewMatrix(cols, rows)
	for k := 0; k < rows; k++ {
		sum := 1.0
		s := make([]float64, cols-1)
		for l := range s {
			v := rng.NormFloat64() * stddev
			sum -= v
			s[l] = v
		}
		index := 0
		for l := 0; l < cols; l++ {
			if k == l {
				transform.Data = append(transform.Data, sum)
			} else {
				transform.Data = append(transform.Data, s[index])
				index++
			}
		}
	}
	return transform
}

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

	rng := rand.New(rand.NewSource(1))
	const Scale = 0.1

	type Network struct {
		Set    tf64.Set
		Others tf64.Set
		L1     tf64.Meta
		L2     tf64.Meta
		Loss   tf64.Meta
		V      tf64.Meta
	}
	var networks []Network
	Width, i := 0, 0
	go camera.Start("/dev/video0")
	for img := range camera.Images {
		if networks == nil {
			width := img.Gray.Bounds().Max.X
			height := img.Gray.Bounds().Max.Y
			Width = width * height
			networks = make([]Network, 3)
			for n := range networks {
				set := tf64.NewSet()
				set.Add("w1", Width, Width-2)
				set.Add("b1", Width-2)
				set.Add("w2", Width-2, Width)
				set.Add("b2", Width)

				for i := range set.Weights {
					w := set.Weights[i]
					if strings.HasPrefix(w.N, "b") {
						w.X = w.X[:cap(w.X)]
						w.States = make([][]float64, StateTotal)
						for i := range w.States {
							w.States[i] = make([]float64, len(w.X))
						}
						continue
					}
					factor := math.Sqrt(float64(w.S[0]))
					for i := 0; i < cap(w.X); i++ {
						w.X = append(w.X, rng.NormFloat64()*factor)
					}
					w.States = make([][]float64, StateTotal)
					for i := range w.States {
						w.States[i] = make([]float64, len(w.X))
					}
				}

				others := tf64.NewSet()
				others.Add("input", Width, Batch)
				others.Add("output", Width, Batch)

				for i := range others.Weights {
					w := others.Weights[i]
					w.X = w.X[:cap(w.X)]
				}

				l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w1"), others.Get("input")), set.Get("b1")))
				l2 := tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2"))
				loss := tf64.Quadratic(l2, others.Get("output"))
				v := tf64.Variance(loss)
				networks[n].Set = set
				networks[n].Others = others
				networks[n].L1 = l1
				networks[n].L2 = l2
				networks[n].Loss = tf64.Avg(loss)
				networks[n].V = v
			}
		}
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		measures := make([]float64, Width)
		for i, v := range img.Gray.Pix {
			measures[i] = float64(v) / 256.0
		}
		network, min := 0, math.MaxFloat64
		for s := 0; s < Batch; s++ {
			transform := MakeRandomTransform(rng, Width, Width, Scale)
			in := NewMatrix(Width, 1, measures...)
			in = transform.MulT(in)
			for n := range networks {
				copy(networks[n].Others.ByName["input"].X[s*Width:(s+1)*Width], in.Data)
				copy(networks[n].Others.ByName["output"].X[s*Width:(s+1)*Width], measures)
			}
		}

		for n := range networks {
			networks[n].Others.Zero()
			networks[n].V(func(a *tf64.V) bool {
				if a.X[0] < min {
					min, network = a.X[0], n
				}
				return true
			})
		}

		networks[network].Others.Zero()

		networks[network].Set.Zero()
		tf64.Gradient(networks[network].Loss)

		norm := 0.0
		for _, p := range networks[network].Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range networks[network].Set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}

		switch network {
		case 0:
			say <- "left"
		case 1:
			say <- "right"
		case 2:
			say <- "straight"
		}
		if i < 1000 {
			i++
		}
	}
}
