package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()
	scanner := bufio.NewScanner(os.Stdin)
	getUserMsg := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}
	agent := NewAgent(&client, getUserMsg)
	err := agent.Launch(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *anthropic.Client, getUserMsg func() (string, bool)) *Agent {
	return &Agent{
		client:     client,
		getUserMsg: getUserMsg,
	}
}

type Agent struct {
	client     *anthropic.Client // anthropic client will look for ANTHROPIC_API_KEY
	getUserMsg func() (string, bool)
}

func (a *Agent) Launch(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}
	fmt.Println("Hello! (exit with <C-c>")

	for {
		// user prompt head
		fmt.Print("User: ")
		userInput, ok := a.getUserMsg()
		if !ok {
			break
		}
		userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMsg)
		msg, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		for _, content := range msg.Content {
			switch content.Type {
			case "text":
				fmt.Printf("Claude: %s\n", content.Text)
			}
		}
	}

	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	msg, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_6,
		MaxTokens: int64(1024),
		Messages:  conversation,
	})
	return msg, err
}
