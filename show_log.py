
a, b = zip(*valosses)
plt.plot(losses, label='train')
plt.plot(a, b, label='avg val per 30')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig(loss_figure)
