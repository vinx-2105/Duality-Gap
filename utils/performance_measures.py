import torch
import torch.optim as optim

def get_current_objective_value(Generator, Discriminator, curr_gen, curr_disc, dataloader2, criterion, device, d_z=100, real_label=1, fake_label=0):
    gen = Generator()
    disc = Discriminator()

    gen.load_state_dict(curr_gen.state_dict())
    disc.load_state_dict(curr_disc.state_dict())

    for i, data in enumerate(dataloader2, 0):
        #calculate the loss over this single batch of data
        real_batch = data.float().to(device)
        b_size = real_batch.size(0)
        label = torch.full((b_size,), real_label, device=device)

        output = disc(real_batch).view(-1)

        errD_real = criterion(output, label)

        noise = torch.randn(b_size, d_z,  device=device)
        # Generate fake image batch with G
        fake = gen(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disc(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        err = errD_real.data.item() + errD_fake.data.item()

        return -1.*err


def get_duality_gap(Generator, Discriminator, curr_gen, curr_disc, global_iter, dataloader1, dataloader2, criterion, device, lrg, lrd, writer,  num_epochs=1, d_z=100, real_label = 1, fake_label=0, beta1=0.5):
    gen = Generator()
    disc = Discriminator()


    gen.load_state_dict(curr_gen.state_dict())
    disc.load_state_dict(curr_disc.state_dict())


    #now evaluate the duality gap by freezing the models
    #(1)train the discriminator first to find disc(worst)
    d_worst = Discriminator()
    d_worst.load_state_dict(disc.state_dict())

    optimizerD = optim.Adam(d_worst.parameters(), lr=lrd, betas=(beta1, 0.999))
    iters=0
    for epoch in range(num_epochs):
        #for each batch in the dataloader
        for i, data in enumerate(dataloader1, 0):
            d_worst.zero_grad()
            ###train with the real batch
            real_batch = data.float().to(device)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = d_worst(real_batch).view(-1)
            # Calculate loss on all-real batch

            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, d_z,  device=device)
            # Generate fake image batch with G
            fake = gen(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = d_worst(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            if iters%500==0:
                print('DG-Discriminator----Global Iter-->{}; Epoch-->{}; Iters-->{}; Error-->{}'.format(global_iter, epoch, iters, errD.item()))
            # Update D
            optimizerD.step()
            iters+=1

    #(2)train the generator first to find gen(worst)
    g_worst = Generator()
    g_worst.load_state_dict(gen.state_dict())

    optimizerG = optim.Adam(g_worst.parameters(), lr=lrg, betas=(beta1, 0.999))

    iters=0
    for epoch in range(num_epochs):
        #for each batch in the dataloader
        for i, data in enumerate(dataloader1, 0):
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            real_batch = data.float().to(device)
            b_size = real_batch.size(0)
            noise = torch.randn(b_size, d_z,  device=device)
            fake = gen(noise)
            output = disc(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            if iters%500==0:
                writer.add_scalar('DG-Generator-Error', errG.item(), global_step=iters)
                print('DG-Generator----Global Iter-->{}; Epoch-->{}; Iters-->{}; Error-->{}'.format(global_iter, epoch, iters, errG.item()))

            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            iters+=1

    M_u_v_worst = get_current_objective_value(Generator, Discriminator, gen, d_worst, dataloader2, criterion, device)
    M_u_worst_v = get_current_objective_value(Generator, Discriminator, g_worst, disc, dataloader2, criterion, device)

    DG = M_u_v_worst - M_u_worst_v

    writer.add_scalar("Duality Gap", DG, global_step=global_iter)

    return DG
